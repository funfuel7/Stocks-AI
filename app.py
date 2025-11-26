# ============================================================
# app.py — NSE Stocks + Options Telegram Bot (Webhook Version)
# Uses Gemini 2.5 Flash for analysis
# Works on Render with: python-telegram-bot[webhooks,job-queue]
# ============================================================

import os
import logging
import asyncio
import datetime as dt
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from google import genai
from google.genai import types as genai_types

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ============================================================
# ENVIRONMENT
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # e.g. https://stocks-ai.onrender.com
PORT = int(os.getenv("PORT", "10000"))

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN missing")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing")
if not WEBHOOK_URL:
    raise RuntimeError("WEBHOOK_URL missing")

client = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("stocks-ai-webhook")

# ============================================================
# SYMBOL MAP
# ============================================================

STOCK_SYMBOL_MAP = {
    "DMART": "DMART.NS",
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "SBIN": "SBIN.NS",
    "AXISBANK": "AXISBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "NIFTYBANK": "^NSEBANK",
}

FUTURES_SCAN_LIST = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK",
    "KOTAKBANK", "TCS", "INFY", "DMART", "NIFTY50", "BANKNIFTY",
]

OPTIONABLE_SYMBOLS = [
    "NIFTY50", "BANKNIFTY", "FINNIFTY",
    "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN",
    "TCS", "INFY", "AXISBANK", "KOTAKBANK", "DMART",
]

# ============================================================
# TIMEFRAME MAPPING
# ============================================================

TIMEFRAME_YF_MAP = {
    "5m": ("5m", "5d"),
    "15m": ("15m", "20d"),
    "1h": ("60m", "60d"),
    "2h": ("60m", "60d"),
    "4h": ("60m", "60d"),
    "6h": ("60m", "60d"),
    "12h": ("60m", "60d"),
    "1d": ("1d", "2y"),
    "daily": ("1d", "2y"),
    "1w": ("1wk", "5y"),
    "weekly": ("1wk", "5y"),
}

DEFAULT_MULTI_TF = ["5m", "15m", "1h", "4h", "1d", "1w"]

# ============================================================
# UTILS
# ============================================================

def map_symbol(symbol_raw: str) -> str:
    s = symbol_raw.strip().upper()
    return STOCK_SYMBOL_MAP.get(s, f"{s}.NS")


def fetch_ohlcv(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    if timeframe not in TIMEFRAME_YF_MAP:
        return None

    interval, period = TIMEFRAME_YF_MAP[timeframe]

    try:
        df = yf.download(
            symbol,
            interval=interval,
            period=period,
            progress=False,
        )
    except Exception as e:
        logger.error(f"yfinance error for {symbol}: {e}")
        return None

    if df is None or df.empty:
        return None

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    ).dropna()

    # Ensure volume is numeric; bad rows become NaN and are dropped
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["volume"])

    if timeframe in ["2h", "4h", "6h", "12h"]:
        rule = {"2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H"}[timeframe]
        df = (
            df.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

    return df

# ============================================================
# INDICATORS (RSI FIX)
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["ema_200"] = df["close"].ewm(span=200).mean()

    # RSI with guaranteed 1D arrays
    delta = df["close"].diff().to_numpy().ravel()
    gain = np.where(delta > 0, delta, 0.0).ravel()
    loss = np.where(delta < 0, -delta, 0.0).ravel()

    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()

    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    df["rsi_14"] = rsi.values

    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["tp"] = tp
    df["vwap"] = (tp * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)

    return df

# ============================================================
# VOLUME PROFILE (HARDENED)
# ============================================================

def compute_volume_profile(df: pd.DataFrame, lookback: int = 120, bins: int = 20):
    """
    Fixed-range volume profile with extra safety:
    - forces tp and volume to numeric
    - replaces any non-numeric with 0
    """
    if len(df) < 10:
        return None

    sub = df.tail(lookback).copy()

    # Force numeric just in case anything weird slipped in
    sub["tp"] = pd.to_numeric(sub["tp"], errors="coerce")
    sub["volume"] = pd.to_numeric(sub["volume"], errors="coerce")
    sub = sub.dropna(subset=["tp", "volume"])

    if sub.empty:
        return None

    low = float(sub["low"].min())
    high = float(sub["high"].max())
    if high <= low:
        return None

    tp = sub["tp"].to_numpy(dtype=float)
    vol = sub["volume"].to_numpy(dtype=float)

    edges = np.linspace(low, high, bins + 1)
    idxs = np.digitize(tp, edges) - 1

    vp: Dict[int, float] = {}
    for idx, v in zip(idxs, vol):
        if 0 <= idx < bins:
            # v is guaranteed float already, no float() conversion
            vp[idx] = vp.get(idx, 0.0) + v

    if not vp:
        return None

    top = sorted(vp.items(), key=lambda x: x[1], reverse=True)[:3]
    nodes: List[float] = []
    for idx, _v in top:
        mid = (edges[idx] + edges[idx + 1]) / 2.0
        nodes.append(round(float(mid), 2))

    return {
        "low": round(low, 2),
        "high": round(high, 2),
        "nodes": nodes,
    }

# ============================================================
# TREND & REVERSALS
# ============================================================

def detect_reversal(df: pd.DataFrame) -> Optional[str]:
    if len(df) < 2:
        return None
    prev, last = df.iloc[-2], df.iloc[-1]

    # Bullish engulfing
    if (
        prev["close"] < prev["open"]
        and last["close"] > last["open"]
        and last["close"] >= prev["open"]
        and last["open"] <= prev["close"]
    ):
        return "bullish_engulfing"

    # Bearish engulfing
    if (
        prev["close"] > prev["open"]
        and last["close"] < last["open"]
        and last["close"] <= prev["open"]
        and last["open"] >= prev["close"]
    ):
        return "bearish_engulfing"

    return None


def trend_label(df: pd.DataFrame) -> str:
    if len(df) < 20:
        return "sideways"

    last = df.iloc[-1]
    price = last["close"]
    ema20, ema50, ema200 = last["ema_20"], last["ema_50"], last["ema_200"]

    recent = df["close"].tail(20)
    slope = (recent.iloc[-1] - recent.iloc[0]) / (recent.iloc[0] + 1e-9)

    if price > ema20 > ema50 > ema200 and slope > 0.02:
        return "strong uptrend"
    if price < ema20 < ema50 < ema200 and slope < -0.02:
        return "strong downtrend"
    if price > ema20 and slope > 0:
        return "mild uptrend"
    if price < ema20 and slope < 0:
        return "mild downtrend"
    return "sideways"

# ============================================================
# MULTI-TF SUMMARY
# ============================================================

def build_multi_tf_summary(symbol: str, yf_symbol: str, timeframe: Optional[str]):
    tfs = [timeframe] if timeframe else DEFAULT_MULTI_TF
    out: Dict[str, Dict[str, Any]] = {}
    err = ""

    for tf in tfs:
        df = fetch_ohlcv(yf_symbol, tf)
        if df is None or df.empty:
            err += f"Failed to fetch {tf} data\n"
            continue

        df = add_indicators(df)
        last = df.iloc[-1]
        vp = compute_volume_profile(df)

        out[tf] = {
            "price": float(last["close"]),
            "ema20": float(last["ema_20"]),
            "ema50": float(last["ema_50"]),
            "ema200": float(last["ema_200"]),
            "rsi": float(last["rsi_14"]),
            "trend": trend_label(df),
            "vp": vp,
        }

    return out, err

# ============================================================
# AUTOSCAN LOGIC
# ============================================================

def autoscan_signal(symbol: str, yf_symbol: str) -> Optional[Dict[str, Any]]:
    df = fetch_ohlcv(yf_symbol, "5m")
    if df is None or len(df) < 50:
        return None

    df = add_indicators(df)
    vp = compute_volume_profile(df, lookback=100, bins=24)
    if not vp:
        return None

    last = df.iloc[-1]
    price = float(last["close"])
    rsi = float(last["rsi_14"])
    trend = trend_label(df)
    pattern = detect_reversal(df)
    nodes = vp["nodes"]

    direction = None
    prob = 0.55

    if "uptrend" in trend:
        direction = "LONG"
        prob += 0.15
        if rsi < 65:
            prob += 0.05
    elif "downtrend" in trend:
        direction = "SHORT"
        prob += 0.15
        if rsi > 35:
            prob += 0.05
    else:
        prob -= 0.05

    if pattern == "bullish_engulfing" and direction != "SHORT":
        direction = "LONG"
        prob += 0.15
    if pattern == "bearish_engulfing" and direction != "LONG":
        direction = "SHORT"
        prob += 0.15

    if not nodes:
        return None

    key = min(nodes, key=lambda x: abs(x - price))
    dist = abs(price - key) / (price + 1e-9) * 100
    if dist < 0.3:
        prob += 0.10

    prob = min(max(prob, 0.0), 0.95)

    if direction == "LONG":
        sl = key * 0.997
        risk = price - sl
        tp = price + 1.9 * risk
    else:
        sl = key * 1.003
        risk = sl - price
        tp = price - 1.9 * risk

    rr = abs(tp - price) / (abs(price - sl) + 1e-9)

    if prob >= 0.85 and rr >= 1.9:
        return {
            "symbol": symbol,
            "direction": direction,
            "prob": prob,
            "entry": round(price, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "rr": rr,
        }
    return None

# ============================================================
# GEMINI HELPERS
# ============================================================

async def gemini_stock(symbol: str, yf_symbol: str, timeframe: Optional[str], data: Dict[str, Dict[str, Any]]) -> str:
    lines = []
    for tf, info in data.items():
        vp = info["vp"]
        if vp:
            vp_txt = f"range {vp['low']}–{vp['high']} nodes={vp['nodes']}"
        else:
            vp_txt = "not available"
        lines.append(
            f"- {tf}: price={info['price']}, rsi={info['rsi']:.1f}, "
            f"trend={info['trend']}, {vp_txt}"
        )

    block = "\n".join(lines)
    tf_label = timeframe if timeframe else "multi-timeframe"

    prompt = f"""
You are an elite Indian stock & derivatives analyst.

Stock: {symbol} (Yahoo: {yf_symbol})
Timeframe focus: {tf_label}

Data:
{block}

Return:

Upside probability: X %
Downside probability: Y %
Flat / choppy probability: Z %

Bias: ...

If highest >= 75% give:

Trade plan:
- Direction:
- Entry:
- Stop loss:
- Take profit 1:
- Take profit 2:
- Approx risk:reward:

Summary (max 3 lines)
"""

    try:
        res = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=700,
                temperature=0.35,
            ),
        )
        return res.text or "Model returned no text."
    except Exception as e:
        logger.error(f"Gemini stock error: {e}")
        return "Gemini stock analysis failed."

async def gemini_options(symbol: str, yf_symbol: str, strike: float, last_price: float) -> str:
    prompt = f"""
You are an expert NSE options risk manager.

Underlying: {symbol} (Yahoo: {yf_symbol})
Last price: {last_price}
Strike: {strike}

Return:

Upside probability: X %
Downside probability: Y %
Flat / choppy probability: Z %

Bias: ...

Options idea:
- Direction: Buy/Sell CE/PE
- Suggested instrument:
- Expiry:
- Underlying reference entry:
- Stop loss:
- Take profit 1:
- Take profit 2:
- Approx risk:reward:

Short explanation (3–5 lines)
"""

    try:
        res = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=700,
                temperature=0.35,
            ),
        )
        return res.text or "Model returned no text."
    except Exception as e:
        logger.error(f"Gemini options error: {e}")
        return "Gemini options analysis failed."

# ============================================================
# TELEGRAM HANDLERS
# ============================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    jq = context.application.job_queue
    chat_id = update.effective_chat.id

    old = context.chat_data.get("scan")
    if old:
        old.schedule_removal()

    job = jq.run_repeating(
        autoscan_job,
        interval=5 * 60,
        first=5,
        data={"chat_id": chat_id, "last": None},
    )
    context.chat_data["scan"] = job

    await update.message.reply_text(
        "🚀 Auto-scan started (5m) on F&O list.\nUse /stop to stop."
    )

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    job = context.chat_data.get("scan")
    if job:
        job.schedule_removal()
        context.chat_data["scan"] = None
        await update.message.reply_text("🛑 Auto-scan stopped.")
    else:
        await update.message.reply_text("No autoscan running.")

async def autoscan_job(context: ContextTypes.DEFAULT_TYPE):
    job_data = context.job.data
    chat_id = job_data["chat_id"]
    now = dt.datetime.utcnow()
    last = job_data.get("last")

    if last and (now - last).total_seconds() < 10 * 60:
        return

    signals: List[Dict[str, Any]] = []
    for s in FUTURES_SCAN_LIST:
        yf_symbol = map_symbol(s)
        try:
            sig = autoscan_signal(s, yf_symbol)
            if sig:
                signals.append(sig)
        except Exception as e:
            logger.error(f"Autoscan error for {s}: {e}")

    if not signals:
        return

    job_data["last"] = now
    context.job.data = job_data

    msg = "⚡ Auto-scan signals:\n\n"
    for s in signals:
        msg += (
            f"*{s['symbol']}* — {s['direction']}\n"
            f"Entry `{s['entry']}` | SL `{s['sl']}` | TP `{s['tp']}`\n"
            f"RR `{s['rr']:.2f}` | Prob `{int(s['prob']*100)}%`\n\n"
        )

    await context.bot.send_message(chat_id, msg, parse_mode="Markdown")

async def cmd_options(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Optionable NSE symbols:\n" + ", ".join(OPTIONABLE_SYMBOLS)
    )

async def cmd_strikeprice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /strikeprice SYMBOL STRIKE")
        return

    symbol = context.args[0].upper()
    try:
        strike = float(context.args[1])
    except ValueError:
        await update.message.reply_text("Strike must be a number, e.g. 25900")
        return

    yf_symbol = map_symbol(symbol)
    df = fetch_ohlcv(yf_symbol, "1d")
    if df is None or df.empty:
        await update.message.reply_text("Could not fetch underlying price.")
        return

    last_price = float(df["close"].iloc[-1])
    await update.message.reply_text("🔍 Analysing options via Gemini...")

    text = await gemini_options(symbol, yf_symbol, strike, last_price)
    await update.message.reply_text(text)

async def cmd_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = update.message.text.strip().split()
    raw = parts[0][1:]  # remove '/'
    symbol = raw.upper()

    timeframe = None
    if len(parts) > 1:
        timeframe = parts[1].lower()
        if timeframe in {"4hr", "4hour"}:
            timeframe = "4h"
        elif timeframe in {"1day", "day"}:
            timeframe = "1d"
        elif timeframe in {"week", "1week"}:
            timeframe = "1w"

    yf_symbol = map_symbol(symbol)

    await update.message.reply_text("📈 Analysing via Gemini...")

    data, err = build_multi_tf_summary(symbol, yf_symbol, timeframe)
    if not data:
        await update.message.reply_text("Failed to build indicators.\n" + err)
        return

    text = await gemini_stock(symbol, yf_symbol, timeframe, data)
    await update.message.reply_text(text)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start — start autoscan\n"
        "/stop — stop autoscan\n"
        "/options — list optionable symbols\n"
        "/strikeprice SYMBOL STRIKE — options analysis\n"
        "/dmart or /dmart 4h — stock analysis\n"
    )

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling update:", exc_info=context.error)

# ============================================================
# MAIN — WEBHOOK
# ============================================================

def main() -> None:
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("stop", cmd_stop))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("options", cmd_options))
    application.add_handler(CommandHandler("strikeprice", cmd_strikeprice))
    application.add_handler(MessageHandler(filters.COMMAND, cmd_stock))

    application.add_error_handler(error_handler)

    logger.info("Starting bot in WEBHOOK mode...")
    application.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=TELEGRAM_BOT_TOKEN,
        webhook_url=f"{WEBHOOK_URL}/{TELEGRAM_BOT_TOKEN}",
        drop_pending_updates=True,
    )

if __name__ == "__main__":
    main()

