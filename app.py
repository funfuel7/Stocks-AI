# ============================================================
# app.py — Indian Stocks + Options AI Bot (OpenRouter version)
#
# - Uses OpenRouter (Llama 3.1 70B) for smart analysis
# - Multi-timeframe analysis: 5m, 15m, 30m, 1h, 2h, 4h, 1d, 3d, 1w
# - Price-action + indicators + fixed-range volume profile
# - Autoscan every 5 minutes with 10 min cooldown, medium frequency
# - Webhook mode for Render
#
# Env required:
#   TELEGRAM_BOT_TOKEN
#   OPENROUTER_API_KEY
#   WEBHOOK_URL   (e.g. https://your-service.onrender.com)
#   PORT          (Render gives this)
# ============================================================

import os
import logging
import asyncio
import datetime as dt
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from openai import OpenAI

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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
PORT = int(os.getenv("PORT", "10000"))

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN missing")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY missing")
if not WEBHOOK_URL:
    raise RuntimeError("WEBHOOK_URL missing")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("stocks-openrouter")

# ============================================================
# SYMBOL MAPS
# ============================================================

STOCK_SYMBOL_MAP: Dict[str, str] = {
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
    "NIFTYBANK": "^NSEBANK",
    "BANKNIFTY": "^NSEBANK",
}

OPTIONABLE_SYMBOLS: List[str] = [
    "NIFTY50", "BANKNIFTY", "FINNIFTY",
    "RELIANCE", "HDFCBANK", "ICICIBANK",
    "SBIN", "TCS", "INFY", "AXISBANK",
    "KOTAKBANK", "DMART",
]

FUTURES_SCAN_LIST: List[str] = [
    "RELIANCE", "HDFCBANK", "ICICIBANK",
    "SBIN", "AXISBANK", "KOTAKBANK",
    "TCS", "INFY", "DMART",
    "NIFTY50", "BANKNIFTY",
]

# ============================================================
# TIMEFRAMES
# ============================================================

# base_interval, period, resample_rule (if we need to aggregate)
TIMEFRAME_CONFIG: Dict[str, Tuple[str, str, Optional[str]]] = {
    "5m": ("5m", "5d", None),
    "15m": ("15m", "20d", None),
    "30m": ("30m", "60d", None),
    "1h": ("60m", "60d", None),
    "2h": ("60m", "60d", "2H"),
    "4h": ("60m", "60d", "4H"),
    "1d": ("1d", "2y", None),
    "3d": ("1d", "2y", "3D"),
    "1w": ("1wk", "5y", None),
}

MULTI_TF_LIST: List[str] = ["5m", "15m", "30m", "1h", "2h", "4h", "1d", "3d", "1w"]

# ============================================================
# UTILS
# ============================================================

def map_symbol(raw: str) -> str:
    s = raw.strip().upper()
    return STOCK_SYMBOL_MAP.get(s, f"{s}.NS")


# ============================================================
# DATA FETCHING (WITH MULTIINDEX + VOLUME FIX)
# ============================================================

def fetch_ohlcv(yf_symbol: str, tf: str) -> Optional[pd.DataFrame]:
    cfg = TIMEFRAME_CONFIG.get(tf)
    if not cfg:
        return None

    interval, period, resample_rule = cfg

    try:
        df = yf.download(
            yf_symbol,
            interval=interval,
            period=period,
            progress=False,
        )
    except Exception as e:
        logger.error(f"yfinance error {yf_symbol} {tf}: {e}")
        return None

    if df is None or df.empty:
        return None

    # Flatten multi-index columns like ('Close', 'DMART.NS')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns]

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

    # Ensure volume numeric
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["volume"])

    if resample_rule:
        df = (
            df.resample(resample_rule)
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
# INDICATORS
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # EMAs
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["ema_200"] = df["close"].ewm(span=200).mean()

    # RSI (14) with 1D arrays
    delta = df["close"].diff().to_numpy().ravel()
    gain = np.where(delta > 0, delta, 0.0).ravel()
    loss = np.where(delta < 0, -delta, 0.0).ravel()
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi_14"] = (100 - (100 / (1 + rs))).values

    # MACD (12,26,9)
    ema_12 = df["close"].ewm(span=12).mean()
    ema_26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    # ATR (14)
    high_low = (df["high"] - df["low"]).abs()
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # Typical price & VWAP
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    df["tp"] = tp
    df["vwap"] = (tp * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)

    return df


# ============================================================
# VOLUME PROFILE
# ============================================================

def compute_volume_profile(df: pd.DataFrame, lookback: int = 150, bins: int = 24):
    if len(df) < 20:
        return None

    sub = df.tail(lookback).copy()
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
            vp[idx] = vp.get(idx, 0.0) + v

    if not vp:
        return None

    top_bins = sorted(vp.items(), key=lambda x: x[1], reverse=True)[:3]
    nodes = []
    for i, _v in top_bins:
        mid = (edges[i] + edges[i + 1]) / 2.0
        nodes.append(round(float(mid), 2))

    return {
        "low": round(low, 2),
        "high": round(high, 2),
        "nodes": nodes,
    }


# ============================================================
# PRICE ACTION HELPERS
# ============================================================

def detect_reversal(df: pd.DataFrame) -> Optional[str]:
    """Simple engulfing / hammer style detection."""
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

    # Basic hammer / shooting star
    body = abs(last["close"] - last["open"])
    full = last["high"] - last["low"] + 1e-9
    upper = last["high"] - max(last["close"], last["open"])
    lower = min(last["close"], last["open"]) - last["low"]

    if body / full < 0.3 and lower / full > 0.5:
        return "bullish_hammer"
    if body / full < 0.3 and upper / full > 0.5:
        return "bearish_shooting_star"

    return None


def trend_label(df: pd.DataFrame) -> str:
    if len(df) < 50:
        return "sideways"

    last = df.iloc[-1]
    price = last["close"]
    ema20, ema50, ema200 = last["ema_20"], last["ema_50"], last["ema_200"]

    recent = df["close"].tail(30)
    slope = (recent.iloc[-1] - recent.iloc[0]) / (recent.iloc[0] + 1e-9)

    if price > ema20 > ema50 > ema200 and slope > 0.02:
        return "strong_uptrend"
    if price < ema20 < ema50 < ema200 and slope < -0.02:
        return "strong_downtrend"
    if price > ema20 and slope > 0:
        return "mild_uptrend"
    if price < ema20 and slope < 0:
        return "mild_downtrend"
    return "sideways"


# ============================================================
# MULTI-TF SUMMARY FOR AI
# ============================================================

def build_multi_tf_summary(symbol: str, yf_symbol: str, timeframe: Optional[str]):
    tfs = [timeframe] if timeframe else MULTI_TF_LIST
    result: Dict[str, Dict[str, Any]] = {}
    err = ""

    for tf in tfs:
        df = fetch_ohlcv(yf_symbol, tf)
        if df is None or df.empty:
            err += f"Failed {tf} data\n"
            continue

        df = add_indicators(df)
        last = df.iloc[-1]
        vp = compute_volume_profile(df)

        result[tf] = {
            "price": float(last["close"]),
            "ema20": float(last["ema_20"]),
            "ema50": float(last["ema_50"]),
            "ema200": float(last["ema_200"]),
            "rsi": float(last["rsi_14"]),
            "macd": float(last["macd"]),
            "macd_signal": float(last["macd_signal"]),
            "atr": float(last["atr_14"]),
            "trend": trend_label(df),
            "reversal": detect_reversal(df),
            "vp": vp,
        }

    return result, err


# ============================================================
# OPENROUTER CHAT WRAPPER
# ============================================================

async def openrouter_chat(prompt: str) -> str:
    try:
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model="meta-llama/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=650,
            temperature=0.25,
        )
        text = completion.choices[0].message.content
        return text or "OpenRouter returned empty text."
    except Exception as e:
        logger.error(f"OpenRouter error: {e}")
        return "OpenRouter request failed."


# ============================================================
# AI: STOCK ANALYSIS
# ============================================================

async def ai_stock_analysis(symbol: str, yf_symbol: str,
                            timeframe: Optional[str],
                            data: Dict[str, Dict[str, Any]]) -> str:
    lines = []
    for tf, info in data.items():
        vp = info["vp"]
        vp_txt = (
            f"VP range {vp['low']}–{vp['high']} nodes={vp['nodes']}"
            if vp else "VP N/A"
        )
        lines.append(
            f"{tf}: price={info['price']}, trend={info['trend']}, "
            f"RSI={info['rsi']:.1f}, MACD={info['macd']:.2f}, "
            f"MACDsig={info['macd_signal']:.2f}, ATR={info['atr']:.2f}, "
            f"reversal={info['reversal']}, {vp_txt}"
        )

    block = "\n".join(lines)

    prompt = f"""
You are a world-class Indian derivatives and price-action trader.

Symbol: {symbol} ({yf_symbol})
Timeframe focus: {"single " + timeframe if timeframe else "multi-timeframe (5m..1w)"}

Multi-timeframe data:
{block}

Use:
- pure price action & structure
- EMA 20/50/200 trend
- RSI signal + divergences
- MACD momentum
- ATR for volatility
- fixed-range volume profile nodes as MAJOR levels
- candlestick reversals near VP nodes for confirmation

Return in MAX 8–10 lines:

1) Upside probability: X %
2) Downside probability: Y %
3) Flat/choppy probability: Z %

4) Bias: Long / Short / Avoid (1 line reason using PA + VP priority)

If highest probability >= 75%:
5) Trade plan:
   - Direction
   - Entry
   - Stoploss (meaningful S/R, not random)
   - TP1
   - TP2
   - Approx RR

6) Final summary (1–2 lines only).
"""

    return await openrouter_chat(prompt)


# ============================================================
# AI: OPTIONS ANALYSIS (SMART, STRIKE OPTIONAL)
# ============================================================

async def ai_options_analysis(symbol: str, yf_symbol: str,
                              strike: Optional[float],
                              last_price: float) -> str:
    today = dt.date.today()

    # Prepare a few upcoming Thursdays for the model to choose
    # (weekly expiries approximation)
    next_thursdays = []
    d = today
    while len(next_thursdays) < 4:
        if d.weekday() == 3:  # Thursday = 3
            next_thursdays.append(d)
        d += dt.timedelta(days=1)

    # Monthly expiry approximation: last Thursday of current or next month
    # (loose approx, still better than nothing)
    def last_thursday_of_month(year, month):
        # start from last day of month and go backwards
        if month == 12:
            nxt = dt.date(year + 1, 1, 1)
        else:
            nxt = dt.date(year, month + 1, 1)
        last_day = nxt - dt.timedelta(days=1)
        d2 = last_day
        while d2.weekday() != 3:
            d2 -= dt.timedelta(days=1)
        return d2

    monthly1 = last_thursday_of_month(today.year, today.month)
    # if already past, go next month
    if monthly1 < today:
        m = today.month + 1
        y = today.year
        if m == 13:
            m = 1
            y += 1
        monthly1 = last_thursday_of_month(y, m)

    weekly_list = [d.strftime("%d %b %Y") for d in next_thursdays]
    monthly_str = monthly1.strftime("%d %b %Y")

    user_strike_info = (
        f"User provided strike: {strike}" if strike is not None
        else "User did not provide any strike; you must choose the safest strike."
    )

    prompt = f"""
You are an expert NSE options trader and risk manager.

Underlying: {symbol} ({yf_symbol})
Spot price: {last_price}
{user_strike_info}

Available nearby weekly expiry dates (Thursday):
{weekly_list}

Recommended monthly expiry candidate (approx last Thursday):
{monthly_str}

Your job:
1. First decide the directional bias: bullish / bearish / flat.
2. Then decide which is LESS RISKY RIGHT NOW:
   - Buy Call (CE)
   - Buy Put (PE)
   - Sell Call (CE)
   - Sell Put (PE)

3. Strike selection:
   - If user strike is clearly reasonable, you MAY use it.
   - But you are allowed to override and choose a BETTER strike
     (ATM or slightly OTM/ITM) if it is safer.
   - Explain if you override the user strike.

4. Expiry selection (SMART — option B):
   - If trend is clean and strong → choose nearest WEEKLY expiry for aggressive directional trades.
   - If current weekly expiry is too close (very little time left) or market is a bit choppy → choose NEXT weekly expiry.
   - If market looks slow / range-bound but biased → choose MONTHLY expiry for safer theta/risk.
   - Always mention the exact date (e.g. 09 Dec 2025) from the provided weekly/monthly list.

Return in MAX 8–10 lines:

1) Upside probability: X %
2) Downside probability: Y %
3) Flat/choppy probability: Z %

4) Suggested options action (risk-first):
   - e.g. "Buy NIFTY50 22200 CE" OR "Sell RELIANCE 2600 PE"
   - clearly mention CE/PE and whether buy or sell.
   - use ATM / slightly OTM strikes as appropriate.

5) Expiry:
   - one of the provided weekly dates OR the monthly date.
   - say: "Choose weekly expiry on <date>" OR "Choose monthly expiry on <date>"

6) Trade plan (for the underlying OR premium level):
   - Entry:
   - Stoploss:
   - Take profit:
   - Approx RR:

7) Short explanation (2–3 lines) linking:
   - price action
   - volume profile / key support-resistance zone
   - why buy vs sell is safer in this scenario.
"""

    return await openrouter_chat(prompt)


# ============================================================
# AUTOSCAN LOGIC
# ============================================================

def autoscan_signal(symbol: str, yf_symbol: str) -> Optional[Dict[str, Any]]:
    df_5m = fetch_ohlcv(yf_symbol, "5m")
    df_1h = fetch_ohlcv(yf_symbol, "1h")
    if df_5m is None or df_1h is None or len(df_5m) < 80:
        return None

    df_5m = add_indicators(df_5m)
    df_1h = add_indicators(df_1h)

    last_5m = df_5m.iloc[-1]
    last_1h = df_1h.iloc[-1]

    vp = compute_volume_profile(df_5m, lookback=150, bins=24)
    if not vp:
        return None

    price = float(last_5m["close"])
    rsi = float(last_5m["rsi_14"])
    trend_5m = trend_label(df_5m)
    trend_1h = trend_label(df_1h)
    pattern = detect_reversal(df_5m)
    nodes = vp["nodes"]

    if not nodes:
        return None

    key = min(nodes, key=lambda x: abs(x - price))
    dist = abs(price - key) / price * 100
    if dist > 0.4:
        return None

    direction = None
    prob = 0.5

    if "uptrend" in trend_1h:
        direction = "LONG"
        prob += 0.2
    elif "downtrend" in trend_1h:
        direction = "SHORT"
        prob += 0.2

    if direction == "LONG" and "uptrend" in trend_5m:
        prob += 0.1
    if direction == "SHORT" and "downtrend" in trend_5m:
        prob += 0.1

    if direction == "LONG" and 35 <= rsi <= 65:
        prob += 0.05
    if direction == "SHORT" and 35 <= rsi <= 65:
        prob += 0.05

    if pattern in ("bullish_engulfing", "bullish_hammer") and direction != "SHORT":
        direction = "LONG"
        prob += 0.2
    if pattern in ("bearish_engulfing", "bearish_shooting_star") and direction != "LONG":
        direction = "SHORT"
        prob += 0.2

    if direction is None:
        return None

    prob = max(0.0, min(prob, 0.98))

    atr = float(last_5m["atr_14"] or 0.0)
    if atr <= 0:
        atr = price * 0.003

    if direction == "LONG":
        sl = min(key - atr, price - 2 * atr)
        risk = price - sl
        tp = price + max(1.9 * risk, 2 * atr)
    else:
        sl = max(key + atr, price + 2 * atr)
        risk = sl - price
        tp = price - max(1.9 * risk, 2 * atr)

    rr = abs(tp - price) / (abs(price - sl) + 1e-9)

    if prob >= 0.85 and rr >= 1.9:
        return {
            "symbol": symbol,
            "direction": direction,
            "entry": round(price, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "rr": rr,
            "prob": prob,
        }

    return None


# ============================================================
# TELEGRAM COMMAND HANDLERS
# ============================================================

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commands:\n"
        "/start — start autoscan (5m, medium frequency)\n"
        "/stop — stop autoscan\n"
        "/options — list optionable symbols\n"
        "/strikeprice SYMBOL [STRIKE] — options AI analysis (strike is optional)\n"
        "/dmart or /dmart 4h — multi-TF AI stock analysis\n"
    )


async def cmd_options(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Optionable NSE symbols:\n" + ", ".join(OPTIONABLE_SYMBOLS)
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    jq = context.application.job_queue
    chat_id = update.effective_chat.id

    old = context.chat_data.get("autoscan_job")
    if old:
        old.schedule_removal()

    job = jq.run_repeating(
        autoscan_job,
        interval=5 * 60,
        first=5,
        data={"chat_id": chat_id, "last_ts": None},
    )
    context.chat_data["autoscan_job"] = job

    await update.message.reply_text(
        "🚀 Auto-scan started.\n"
        "Every 5 min I scan F&O list using PA + VP.\n"
        "I only send signals when probability ≥ 85% and RR ≥ 1.9.\n"
        "Use /stop to stop."
    )


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    job = context.chat_data.get("autoscan_job")
    if job:
        job.schedule_removal()
        context.chat_data["autoscan_job"] = None
        await update.message.reply_text("🛑 Auto-scan stopped.")
    else:
        await update.message.reply_text("No autoscan running.")


async def autoscan_job(context: ContextTypes.DEFAULT_TYPE):
    data = context.job.data
    chat_id = data["chat_id"]
    now = dt.datetime.utcnow()
    last_ts = data.get("last_ts")

    if last_ts and (now - last_ts).total_seconds() < 600:
        return

    signals: List[Dict[str, Any]] = []

    for s in FUTURES_SCAN_LIST:
        yf_symbol = map_symbol(s)
        try:
            sig = autoscan_signal(s, yf_symbol)
            if sig:
                signals.append(sig)
        except Exception as e:
            logger.error(f"autoscan error {s}: {e}")

    if not signals:
        return

    data["last_ts"] = now

    msg = "⚡ Auto-scan signals:\n\n"
    for s in signals:
        msg += (
            f"*{s['symbol']}* — {s['direction']}\n"
            f"Entry `{s['entry']}` | SL `{s['sl']}` | TP `{s['tp']}`\n"
            f"RR `{s['rr']:.2f}` | Prob `{int(s['prob']*100)}%`\n\n"
        )

    await context.bot.send_message(chat_id, msg, parse_mode="Markdown")


async def cmd_strikeprice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 1:
        await update.message.reply_text("Usage: /strikeprice SYMBOL [STRIKE]")
        return

    symbol = args[0].upper()
    strike: Optional[float] = None

    if len(args) >= 2:
        try:
            strike = float(args[1])
        except ValueError:
            # If user gave bad strike, just ignore and let AI choose
            strike = None

    yf_symbol = map_symbol(symbol)
    df = fetch_ohlcv(yf_symbol, "1d")
    if df is None or df.empty:
        await update.message.reply_text("Could not fetch underlying price.")
        return

    last_price = float(df["close"].iloc[-1])
    await update.message.reply_text("📉 Analysing options via OpenRouter…")

    text = await ai_options_analysis(symbol, yf_symbol, strike, last_price)
    await update.message.reply_text(text)


async def cmd_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    parts = update.message.text.strip().split()
    symbol = parts[0][1:].upper()

    tf = None
    if len(parts) > 1:
        tf = parts[1].lower()
        if tf in {"4hr", "4hour"}:
            tf = "4h"
        if tf in {"1day", "day"}:
            tf = "1d"
        if tf in {"1week", "week"}:
            tf = "1w"

    yf_symbol = map_symbol(symbol)

    await update.message.reply_text("📊 Analysing with multi-timeframe AI…")

    data, err = build_multi_tf_summary(symbol, yf_symbol, tf)
    if not data:
        await update.message.reply_text("Failed to build indicators.\n" + err)
        return

    text = await ai_stock_analysis(symbol, yf_symbol, tf, data)
    await update.message.reply_text(text)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Error:", exc_info=context.error)


# ============================================================
# MAIN — WEBHOOK
# ============================================================

def main() -> None:
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("options", cmd_options))
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("strikeprice", cmd_strikeprice))
    app.add_handler(MessageHandler(filters.COMMAND, cmd_stock))

    app.add_error_handler(error_handler)

    logger.info("Starting OpenRouter stocks bot (webhook mode)…")

    app.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=TELEGRAM_BOT_TOKEN,
        webhook_url=f"{WEBHOOK_URL}/{TELEGRAM_BOT_TOKEN}",
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
