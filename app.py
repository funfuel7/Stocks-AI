# ============================================================
# app.py — Telegram webhook bot + Gemini 2.5 Flash + Indian stocks
#
# - Runs as a WEBHOOK on Render (must listen on PORT).
# - /start: starts autoscan (5m) on NSE F&O universe.
# - /stop: stops autoscan.
# - /options: list F&O stocks/indexes usable for options.
# - /strikeprice <symbol> <strike>: options analysis (Gemini).
# - /<symbol> [tf]: manual stock analysis using Gemini.
#
# IMPORTANT:
#   Set WEBHOOK_URL env var on Render.
#   Example: https://stocks-ai.onrender.com
#
# ============================================================

import os
import logging
import math
import asyncio
import datetime as dt
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from google import genai
from google.genai import types as genai_types


# ============================================================
# ENV
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0") or "0")
PORT = int(os.getenv("PORT", "10000"))

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN missing")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing")

if not WEBHOOK_URL:
    raise RuntimeError("WEBHOOK_URL missing")

client = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================
# LOG
# ============================================================

logging.basicConfig(
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("stocks-ai")

# ============================================================
# SYMBOL MAP (NSE)
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
    "HINDUNILVR": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "NIFTY50": "^NSEI",
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "NIFTYBANK": "^NSEBANK",
}

FUTURES_SCAN_LIST = [
    "RELIANCE",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "AXISBANK",
    "KOTAKBANK",
    "TCS",
    "INFY",
    "DMART",
    "NIFTY50",
    "BANKNIFTY",
]

OPTIONABLE_SYMBOLS = [
    "NIFTY50",
    "BANKNIFTY",
    "FINNIFTY",
    "RELIANCE",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "TCS",
    "INFY",
    "AXISBANK",
    "KOTAKBANK",
    "DMART",
]

# ============================================================
# TIMEFRAMES
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

def map_user_symbol(cmd: str) -> str:
    c = cmd.upper()
    return STOCK_SYMBOL_MAP.get(c, f"{c}.NS")


def fetch_ohlcv(yf_symbol: str, tf: str) -> Optional[pd.DataFrame]:
    if tf not in TIMEFRAME_YF_MAP:
        return None

    interval, period = TIMEFRAME_YF_MAP[tf]

    try:
        df = yf.download(
            yf_symbol, interval=interval, period=period, progress=False
        )
    except Exception:
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

    if tf in ["2h", "4h", "6h", "12h"]:
        rule = {"2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H"}[tf]
        df = (
            df.resample(rule)
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna()
        )

    return df


# ============================================================
# FIXED RSI INDICATOR
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["ema_200"] = df["close"].ewm(span=200).mean()

    # --- FIX: ensure gain/loss are 1-D arrays
    delta = df["close"].diff().to_numpy().ravel()
    gain = np.where(delta > 0, delta, 0.0).ravel()
    loss = np.where(delta < 0, -delta, 0.0).ravel()

    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi_14"] = (100 - (100 / (1 + rs))).values

    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["tp"] = tp
    df["vwap"] = (tp * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)
    return df


# ============================================================
# VOLUME PROFILE
# ============================================================

def compute_volume_profile(df, lookback=120, bins=20):
    if len(df) < 10:
        return None

    sub = df.tail(lookback)
    low = float(sub["low"].min())
    high = float(sub["high"].max())
    if high <= low:
        return None

    tp = sub["tp"]
    vol = sub["volume"]

    edges = np.linspace(low, high, bins + 1)
    idxs = np.digitize(tp, edges) - 1

    vp = {}
    for i, v in zip(idxs, vol):
        if 0 <= i < bins:
            vp[i] = vp.get(i, 0) + float(v)

    if not vp:
        return None

    top_nodes = sorted(vp.items(), key=lambda x: x[1], reverse=True)[:3]
    nodes = [(edges[i] + edges[i+1]) / 2 for i, _ in top_nodes]
    return {"low": round(low, 2), "high": round(high, 2), "nodes": [round(x, 2) for x in nodes]}


# ============================================================
# REVERSALS + TREND
# ============================================================

def detect_reversal_pattern(df):
    if len(df) < 2:
        return None
    prev, last = df.iloc[-2], df.iloc[-1]

    if (
        prev["close"] < prev["open"]
        and last["close"] > last["open"]
        and last["close"] >= prev["open"]
        and last["open"] <= prev["close"]
    ):
        return "bullish_engulfing"

    if (
        prev["close"] > prev["open"]
        and last["close"] < last["open"]
        and last["close"] <= prev["open"]
        and last["open"] >= prev["close"]
    ):
        return "bearish_engulfing"
    return None


def trend_label(df):
    if len(df) < 20:
        return "sideways"
    last = df.iloc[-1]
    p = last["close"]
    ema20, ema50, ema200 = last["ema_20"], last["ema_50"], last["ema_200"]

    recent = df["close"].tail(20)
    slope = (recent.iloc[-1] - recent.iloc[0]) / (recent.iloc[0] + 1e-9)

    if p > ema20 > ema50 > ema200 and slope > 0.02:
        return "strong uptrend"
    if p < ema20 < ema50 < ema200 and slope < -0.02:
        return "strong downtrend"
    if p > ema20 and slope > 0:
        return "mild uptrend"
    if p < ema20 and slope < 0:
        return "mild downtrend"
    return "sideways"


# ============================================================
# MULTI TF SUMMARY
# ============================================================

def build_multi_tf_summary(symbol, yf_symbol, timeframe=None):
    tfs = [timeframe] if timeframe else DEFAULT_MULTI_TF
    data = {}
    err = ""

    for tf in tfs:
        df = fetch_ohlcv(yf_symbol, tf)
        if df is None or df.empty:
            err += f"Could not fetch {tf} data\n"
            continue

        df = add_indicators(df)
        last = df.iloc[-1]
        vp = compute_volume_profile(df)

        data[tf] = {
            "current_price": float(last["close"]),
            "ema_20": float(last["ema_20"]),
            "ema_50": float(last["ema_50"]),
            "ema_200": float(last["ema_200"]),
            "rsi": float(last["rsi_14"]),
            "trend": trend_label(df),
            "volume_profile": vp,
        }

    return data, err


# ============================================================
# AUTOSCAN
# ============================================================

def build_autoscan_signal(symbol, yf_symbol):
    df = fetch_ohlcv(yf_symbol, "5m")
    if df is None or len(df) < 50:
        return None

    df = add_indicators(df)
    vp = compute_volume_profile(df, lookback=100, bins=24)
    if vp is None:
        return None

    last = df.iloc[-1]
    price = float(last["close"])
    rsi = float(last["rsi_14"])
    trend = trend_label(df)
    pattern = detect_reversal_pattern(df)
    nodes = vp.get("nodes", [])

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

    if pattern == "bullish_engulfing":
        if direction != "SHORT":
            direction = "LONG"
            prob += 0.15
    if pattern == "bearish_engulfing":
        if direction != "LONG":
            direction = "SHORT"
            prob += 0.15

    if not nodes:
        return None

    node = min(nodes, key=lambda x: abs(x - price))
    dist = abs(price - node) / (price + 1e-9) * 100

    if dist < 0.3:
        prob += 0.10

    prob = min(0.95, max(0.0, prob))

    if direction == "LONG":
        sl = node * 0.997
        risk = price - sl
        tp = price + risk * 1.9
    else:
        sl = node * 1.003
        risk = sl - price
        tp = price - risk * 1.9

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
# GEMINI HELPERS (STOCK + OPTIONS)
# ============================================================

async def gemini_analyze_stock(symbol, yf_symbol, timeframe, multi_tf):
    lines = []
    for tf, info in multi_tf.items():
        vp = info["volume_profile"]
        vp_txt = (
            f"range {vp['low']}–{vp['high']} nodes={vp['nodes']}"
            if vp else "not available"
        )
        lines.append(
            f"- {tf}: price={info['current_price']}, trend={info['trend']}, "
            f"RSI≈{info['rsi']:.1f}, EMA20={info['ema_20']}, EMA50={info['ema_50']}, "
            f"EMA200={info['ema_200']}, VP: {vp_txt}"
        )

    data_block = "\n".join(lines)

    tf_label = timeframe if timeframe else "multi-timeframe"

    prompt = f"""
You are an elite Indian stock analyst.

Stock: {symbol}
Timeframe: {tf_label}

Data:
{data_block}

Format:

Upside probability: X %
Downside probability: Y %
Flat / choppy probability: Z %

Bias: ...

If highest >= 75% → give:

Trade plan:
- Direction: Long/Short
- Entry: ...
- Stop loss: ...
- Take profit 1: ...
- Take profit 2: ...
- Approx risk:reward: ...

Summary (3 lines max)
"""

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


async def gemini_analyze_options(symbol, yf_symbol, strike, last_price):
    prompt = f"""
You are an elite Indian NSE options analyst.

Underlying: {symbol}
Last price: {last_price}
Strike: {strike}

Give:
Upside probability: X %
Downside probability: Y %
Flat / choppy probability: Z %

Bias: ...

Trade idea:
- Direction: Buy CE/Buy PE/Sell CE/Sell PE
- Suggested instrument: ...
- Expiry: ...
- Underlying ref entry: ...
- Stop loss: ...
- Take profit 1:
- Take profit 2:
- Approx risk:reward:

Short explanation (3–5 lines)
"""

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


# ============================================================
# TELEGRAM HANDLERS
# ============================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    jq = context.application.job_queue

    if jq is None:
        await update.message.reply_text("JobQueue unavailable.")
        return

    old = context.chat_data.get("autoscan")
    if old:
        old.schedule_removal()

    job = jq.run_repeating(
        autoscan_job,
        interval=5 * 60,
        first=5,
        data={"chat_id": chat_id, "last_signal_time": None},
    )
    context.chat_data["autoscan"] = job

    await update.message.reply_text(
        "🚀 Auto-scan started (5m). Will scan NSE F&O every 5 min.\n"
        "Use /stop to disable."
    )


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    j = context.chat_data.get("autoscan")
    if j:
        j.schedule_removal()
        context.chat_data["autoscan"] = None
        await update.message.reply_text("🛑 Auto-scan stopped.")
    else:
        await update.message.reply_text("No autoscan running.")


async def autoscan_job(context: ContextTypes.DEFAULT_TYPE):
    dt_now = dt.datetime.utcnow()
    job_data = context.job.data
    last = job_data.get("last_signal_time")

    if last and (dt_now - last).total_seconds() < 10 * 60:
        return

    chat_id = job_data["chat_id"]
    signals = []

    for s in FUTURES_SCAN_LIST:
        yf_symbol = map_user_symbol(s)
        try:
            sig = build_autoscan_signal(s, yf_symbol)
            if sig:
                signals.append(sig)
        except Exception as e:
            logger.exception(e)

    if not signals:
        return

    job_data["last_signal_time"] = dt_now
    context.job.data = job_data

    msg = "⚡ Auto-scan signals:\n\n"
    for s in signals:
        msg += (
            f"*{s['symbol']}* — {s['direction']}\n"
            f"Entry `{s['entry']}` | SL `{s['sl']}` | TP `{s['tp']}`\n"
            f"RR `{s['rr']:.2f}`  Prob `{int(s['prob']*100)}%`\n\n"
        )

    await context.bot.send_message(chat_id, msg, parse_mode="Markdown")


async def options_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Optionable symbols:\n" + ", ".join(OPTIONABLE_SYMBOLS)
    )


async def strikeprice_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "Usage: /strikeprice <symbol> <strike>"
        )
        return

    sym = args[0].upper()
    strike = float(args[1])
    yf_symbol = map_user_symbol(sym)

    df = fetch_ohlcv(yf_symbol, "1d")
    if df is None:
        await update.message.reply_text("Error fetching underlying.")
        return

    last = float(df["close"].iloc[-1])
    await update.message.reply_text("🔍 Analysing options via Gemini...")

    try:
        text = await gemini_analyze_options(sym, yf_symbol, strike, last)
        await update.message.reply_text(text)
    except Exception:
        await update.message.reply_text("Gemini error. Try later.")


async def generic_stock_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = update.message.text.strip().split()
    cmd = parts[0][1:].upper()
    tf = parts[1].lower() if len(parts) > 1 else None

    yf_symbol = map_user_symbol(cmd)

    await update.message.reply_text("📈 Analysing via Gemini...")

    mdata, err = build_multi_tf_summary(cmd, yf_symbol, tf)
    if not mdata:
        await update.message.reply_text("Error fetching data.")
        return

    try:
        txt = await gemini_analyze_stock(cmd, yf_symbol, tf, mdata)
        await update.message.reply_text(txt)
    except Exception:
        await update.message.reply_text("Gemini error. Try later.")


async def help_cmd(update: Update, context):
    await update.message.reply_text(
        "/start — start autoscan\n"
        "/stop — stop autoscan\n"
        "/options — list options symbols\n"
        "/strikeprice SYMBOL STRIKE\n"
        "/dmart or /dmart 4h — analysis"
    )


async def error_handler(update, context):
    logger.error("Error:", exc_info=context.error)


# ============================================================
# MAIN — WEBHOOK
# ============================================================

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("options", options_cmd))
    app.add_handler(CommandHandler("strikeprice", strikeprice_cmd))
    app.add_handler(MessageHandler(filters.COMMAND, generic_stock_cmd))

    app.add_error_handler(error_handler)

    logger.info("Running webhook…")
    app.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=TELEGRAM_BOT_TOKEN,
        webhook_url=f"{WEBHOOK_URL}/{TELEGRAM_BOT_TOKEN}",
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
