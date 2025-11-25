# ============================================================
# app.py — Telegram bot + Gemini 2.5 Flash + Indian stocks
#
# Features:
# - /start → starts autoscan (every 5 min) of F&O stocks (NSE) using indicators
#   and fixed-range volume profile style logic; sends only high-probability
#   intraday signals (prob ≥ 85%, RR ≥ 1.9).
# - /stop → stops autoscan.
# - /options → lists common F&O stocks/indices for options trading.
# - /strikeprice <symbol> <strike> → options view (CE/PE bias, expiry idea,
#   TP, SL, probability).
# - /DMART or /dmart or `/dmart 4h` → Gemini-based analysis with probabilities
#   (Upside / Downside / Flat), and if max prob ≥ 75%, gives entry/SL/TP.
#
# Deploy as a Render Web Service (start command: python app.py).
# ============================================================

import os
import logging
import re
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

# New Google Gen AI SDK (Gemini 2.x)
# Docs: https://ai.google.dev/gemini-api/docs/quickstart
from google import genai
from google.genai import types as genai_types

# ============================================================
# ENVIRONMENT
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0") or "0")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN env var is required")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY env var is required")

client = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("stock-gemini-bot")

# ============================================================
# SYMBOL CONFIG (NSE)
# ============================================================

# Map user-friendly commands to Yahoo Finance symbols.
# You can extend this list as you like.
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
    "LT": "LT.NS",
    "ITC": "ITC.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "NIFTYBANK": "^NSEBANK",
    "BANKNIFTY": "^NSEBANK",
}

# F&O universe for autoscan (you can add more).
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

# Optionable symbols (for /options list)
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
# TIMEFRAME CONFIG
# ============================================================

# Map user timeframe to (yfinance interval, yfinance period)
TIMEFRAME_YF_MAP = {
    "5m": ("5m", "5d"),
    "15m": ("15m", "20d"),
    "1h": ("60m", "60d"),
    # 2h, 4h, 6h, 12h are approximated from 60m data (resampled in analysis)
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
# UTILITIES
# ============================================================


def map_user_symbol(symbol_raw: str) -> Optional[str]:
    """Map user command like 'dmart' to Yahoo ticker like 'DMART.NS'."""
    s = symbol_raw.strip().upper()
    if s in STOCK_SYMBOL_MAP:
        return STOCK_SYMBOL_MAP[s]
    # Fallback: assume NSE stock with .NS suffix
    return f"{s}.NS"


def fetch_ohlcv(
    yf_symbol: str, timeframe: str
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from Yahoo Finance for given timeframe."""
    tf = timeframe.lower()
    if tf not in TIMEFRAME_YF_MAP:
        return None

    interval, period = TIMEFRAME_YF_MAP[tf]
    try:
        df = yf.download(
            yf_symbol,
            interval=interval,
            period=period,
            progress=False,
        )
    except Exception as e:
        logger.exception(f"Error fetching data for {yf_symbol}: {e}")
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
    )
    df = df.dropna()

    # Resample for 2h, 4h, 6h, 12h if needed
    if tf in ["2h", "4h", "6h", "12h"]:
        rule = {"2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H"}[tf]
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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA, RSI, VWAP style indicators."""
    df = df.copy()

    # EMAs
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI 14
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Typical price & VWAP-like measure
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    df["tp"] = tp
    df["vwap"] = (tp * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-9)

    return df


def compute_volume_profile(
    df: pd.DataFrame, lookback: int = 120, bins: int = 20
) -> Optional[Dict[str, Any]]:
    """Approx fixed range volume profile summary between recent swing low & high."""
    if len(df) < 10:
        return None

    sub = df.tail(lookback).copy()
    low = float(sub["low"].min())
    high = float(sub["high"].max())
    if high <= low:
        return None

    tp = sub["tp"]
    vol = sub["volume"]

    bin_edges = np.linspace(low, high, bins + 1)
    indices = np.digitize(tp, bin_edges) - 1

    vp = {}
    for idx, v in zip(indices, vol):
        if 0 <= idx < bins:
            vp[idx] = vp.get(idx, 0) + float(v)

    if not vp:
        return None

    top_nodes = sorted(vp.items(), key=lambda x: x[1], reverse=True)[:3]
    nodes = []
    for idx, v in top_nodes:
        mid = (bin_edges[idx] + bin_edges[idx + 1]) / 2.0
        nodes.append(round(float(mid), 2))

    return {
        "low": round(low, 2),
        "high": round(high, 2),
        "nodes": nodes,
    }


def detect_reversal_candle(last: pd.Series) -> Optional[str]:
    """Simple candlestick reversal detection for autoscan."""
    o = last["open"]
    h = last["high"]
    l = last["low"]
    c = last["close"]

    body = abs(c - o)
    range_ = h - l + 1e-9

    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l

    # Basic hammer / bullish pin
    if (lower_shadow > 2 * body) and (upper_shadow < body) and (c > o):
        return "bullish_hammer"

    # Basic shooting star / bearish pin
    if (upper_shadow > 2 * body) and (lower_shadow < body) and (c < o):
        return "bearish_shooting_star"

    # Bullish engulfing
    # For this we need previous candle; can't here (we use this only when we already have previous)
    return None


def detect_reversal_pattern(df: pd.DataFrame) -> Optional[str]:
    """Use last two candles to detect reversal pattern for autoscan."""
    if len(df) < 2:
        return None

    last2 = df.tail(2)
    prev = last2.iloc[0]
    last = last2.iloc[1]

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

    # Single-candle patterns
    pattern = detect_reversal_candle(last)
    return pattern


# ============================================================
# GEMINI HELPERS
# ============================================================


async def gemini_analyze_stock(
    symbol: str,
    yf_symbol: str,
    timeframe: Optional[str],
    multi_tf_data: Dict[str, Dict[str, Any]],
) -> str:
    """
    Ask Gemini to act as elite Indian market + risk management expert.
    `multi_tf_data` is a dict like:
    {
      "5m": {
         "current_price": ...,
         "trend": "uptrend"/"downtrend"/"sideways",
         "rsi": ...,
         "ema_20": ...,
         "ema_50": ...,
         "ema_200": ...,
         "volume_profile": {low, high, nodes:[...]}
      },
      ...
    }
    """
    tf_label = timeframe if timeframe else "multi-timeframe (5m..weekly)"

    # Build structured text summary for the model
    lines = []
    for tf, info in multi_tf_data.items():
        vp = info.get("volume_profile")
        vp_txt = (
            f"range {vp['low']}–{vp['high']}, high-volume nodes {vp['nodes']}"
            if vp
            else "not available"
        )
        lines.append(
            f"- {tf}: price={info['current_price']:.2f}, trend={info['trend']}, "
            f"RSI≈{info['rsi']:.1f}, EMA20={info['ema_20']:.2f}, "
            f"EMA50={info['ema_50']:.2f}, EMA200={info['ema_200']:.2f}, "
            f"fixed-range volume profile: {vp_txt}"
        )

    data_block = "\n".join(lines)

    if timeframe:
        instruction_scope = (
            f"Focus ONLY on the {timeframe} timeframe for final signal."
        )
    else:
        instruction_scope = (
            "Use all timeframes together; for final probabilities, consider the "
            "overall structure from intraday (5m/15m/1h/4h) plus higher timeframes "
            "(daily/weekly)."
        )

    prompt = f"""
You are an Indian equity & derivatives expert, risk manager in the top 1% of traders worldwide.
You analyse NSE stocks and derivatives with extreme discipline and risk management.

User wants to trade stock: {symbol} (Yahoo: {yf_symbol})
Timeframe focus: {tf_label}

Data summary (computed from recent OHLCV):
{data_block}

Your task:

1. Analyse the stock like a professional Indian market analyst & futures/options trader.
2. Consider ALL of these:
   - Overall trend on each timeframe.
   - Fixed-range volume profile between recent swing high & low (high-volume nodes are key S/R).
   - Candlestick behaviour (reversals, strong trend candles).
   - Momentum (RSI style signal).
   - Confluence between intraday & higher timeframes.

3. Output the following sections EXACTLY in this textual format (no JSON):

Upside probability: X %
Downside probability: Y %
Flat / choppy probability: Z %

Bias: (choose one: Upside / Downside / Flat)

If the highest probability is at least 75%, then ALSO give:
- Direction: Long or Short
- Entry zone: (a tight price range, e.g. 4100–4125)
- Stop loss: (strong S/R that is not easily hunted, NOT random)
- Take profit 1:
- Take profit 2:
- Approx risk:reward: (e.g. 1:2.3)

Then add a VERY SHORT summary (2–4 lines max) explaining:
- Why that move is likely (trend + volume profile + candles).
- How risk is managed (where SL is, why it is at a protected level).
- If probability < 75% for both up and down, clearly say: "Better to AVOID this stock for now and wait."

Be concise, no fluff. Give numbers that look realistic for Indian markets.
"""

    # Use Gemini 2.5 Flash with JSON-style constraints if needed; here text is fine.
    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-2.5-flash",
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            max_output_tokens=700,
            temperature=0.4,
        ),
    )

    text = response.text or "Model returned no text."
    return text


async def gemini_analyze_options(
    symbol: str,
    yf_symbol: str,
    strike_price: float,
    last_price: float,
) -> str:
    """
    Options view using Gemini: suggest CE/PE, buy/sell, expiry, TP/SL, probabilities.
    This is *conceptual* risk-aware guidance, based on underlying level vs strike.
    """
    distance = (strike_price - last_price) / (last_price + 1e-9) * 100

    prompt = f"""
You are an Indian options and risk management expert in the top 1% of traders.
You specialise in NSE index & stock options: NIFTY, BANKNIFTY, large-cap F&O stocks.

Underlying: {symbol} (Yahoo: {yf_symbol})
Last traded price (approx from recent OHLCV): {last_price:.2f}
User is asking about strike price: {strike_price:.2f}
Relative distance from underlying ≈ {distance:.2f} %.

Task:

1. First, analyse the underlying price behaviour and say:
   - Upside probability: X %
   - Downside probability: Y %
   - Flat / choppy probability: Z %
   Pick realistic values that sum to 100%.

2. Based on that, decide:
   - Whether user should prefer Buying options (CE/PE) OR Selling options (CE/PE).
   - Which side: CE or PE.
   - Whether the given strike is OTM/ATM/ITM; if it is too far OTM, suggest a more reasonable strike.

3. Risk management is the highest priority:
   - Suggest ONE main trade idea only.
   - Include:
     - Direction: Buy CE / Buy PE / Sell CE / Sell PE (ONE only)
     - Underlying entry reference: (range near current price)
     - Suggested strike: <symbol + strike + CE/PE>
     - Expiry: choose a safe expiry (e.g. nearest weekly IF intraday, otherwise monthly) and explain why.
     - Stop loss logic: either on underlying price OR option premium, but explain clearly.
     - Take profit levels: TP1 and TP2.
     - Approx risk:reward.

4. If the probabilities show that the market is likely flat/choppy, clearly say:
   - "Avoid option trade; risk is not favourable."

5. Format response EXACTLY:

Upside probability: X %
Downside probability: Y %
Flat / choppy probability: Z %

Bias: Upside / Downside / Flat

Trade idea:
- Direction: ...
- Suggested instrument: <OPTION SYMBOL, e.g. NIFTY 25900 CE>
- Expiry: <weekly or monthly, date>
- Underlying reference entry: ...
- Stop loss: ...
- Take profit 1: ...
- Take profit 2: ...
- Approx risk:reward: 1:R

Short explanation (3–5 lines):
- Why this direction
- Why this strike & expiry
- How risk is controlled
"""

    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-2.5-flash",
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            max_output_tokens=700,
            temperature=0.4,
        ),
    )
    return response.text or "Model returned no text."


# ============================================================
# ANALYSIS HELPERS
# ============================================================


def trend_label(df: pd.DataFrame) -> str:
    """Very rough trend detection using EMA and slope."""
    if len(df) < 20:
        return "sideways"
    last = df.iloc[-1]
    ema20 = last["ema_20"]
    ema50 = last["ema_50"]
    ema200 = last["ema_200"]
    price = last["close"]

    # Slope via difference over last 20 bars
    recent = df.tail(20)
    slope = (recent["close"].iloc[-1] - recent["close"].iloc[0]) / (
        recent["close"].iloc[0] + 1e-9
    )

    if price > ema20 > ema50 > ema200 and slope > 0.02:
        return "strong uptrend"
    if price < ema20 < ema50 < ema200 and slope < -0.02:
        return "strong downtrend"
    if price > ema20 and slope > 0:
        return "mild uptrend"
    if price < ema20 and slope < 0:
        return "mild downtrend"
    return "sideways"


def build_multi_tf_summary(
    symbol: str, yf_symbol: str, timeframe: Optional[str] = None
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """
    Fetch OHLCV and compute indicators for either:
    - single timeframe, if timeframe is given
    - multiple timeframes otherwise
    Returns: (multi_tf_data, error_message)
    """
    tf_list = [timeframe] if timeframe else DEFAULT_MULTI_TF
    data: Dict[str, Dict[str, Any]] = {}
    error_msg = ""

    for tf in tf_list:
        df = fetch_ohlcv(yf_symbol, tf)
        if df is None or df.empty:
            error_msg += f"Could not fetch data for {symbol} on timeframe {tf}.\n"
            continue

        df = add_indicators(df)
        last = df.iloc[-1]
        vp = compute_volume_profile(df)

        info = {
            "current_price": float(last["close"]),
            "ema_20": float(last["ema_20"]),
            "ema_50": float(last["ema_50"]),
            "ema_200": float(last["ema_200"]),
            "rsi": float(last["rsi_14"]),
            "trend": trend_label(df),
            "volume_profile": vp,
        }
        data[tf] = info

    return data, error_msg


# ============================================================
# AUTOSCAN LOGIC (NO GEMINI, INDICATOR-BASED)
# ============================================================


def build_autoscan_signal(
    symbol: str, yf_symbol: str
) -> Optional[Dict[str, Any]]:
    """
    For autoscan:
    - Use 5m timeframe.
    - Compute FR volume profile + reversal pattern + RSI/EMA trend.
    - Return signal dict if probability ≥ 0.85 and RR ≥ 1.9, else None.
    """
    df = fetch_ohlcv(yf_symbol, "5m")
    if df is None or len(df) < 50:
        return None

    df = add_indicators(df)
    vp = compute_volume_profile(df, lookback=100, bins=24)
    if vp is None:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    current_price = float(last["close"])
    rsi = float(last["rsi_14"])
    ema20 = float(last["ema_20"])
    ema50 = float(last["ema_50"])

    pattern = detect_reversal_pattern(df)
    trend = trend_label(df)

    direction = None
    base_prob = 0.55

    # Trend-based tilt
    if "uptrend" in trend:
        base_prob += 0.10
        if rsi < 65:
            base_prob += 0.05
        direction = "LONG"
    elif "downtrend" in trend:
        base_prob += 0.10
        if rsi > 35:
            base_prob += 0.05
        direction = "SHORT"
    else:
        # sideways: only take strong reversal patterns
        base_prob -= 0.05

    # Pattern-based boost
    if pattern in ["bullish_engulfing", "bullish_hammer"]:
        if direction is None:
            direction = "LONG"
        if direction == "LONG":
            base_prob += 0.15
    elif pattern in ["bearish_engulfing", "bearish_shooting_star"]:
        if direction is None:
            direction = "SHORT"
        if direction == "SHORT":
            base_prob += 0.15

    # Volume profile confluence:
    # if price near a high-volume node, treat it as S/R
    vp_nodes = vp.get("nodes", [])
    if vp_nodes:
        nearest_node = min(vp_nodes, key=lambda x: abs(current_price - x))
        dist = abs(current_price - nearest_node) / (current_price + 1e-9) * 100
        if dist < 0.3:
            base_prob += 0.1  # confluence of FRVP
        elif dist > 1.5:
            base_prob -= 0.05

    prob = max(0.0, min(base_prob, 0.95))  # clamp between 0..0.95

    # Build SL and TP around strong S/R
    # Use nearest volume node as key level
    if not vp_nodes:
        return None

    nearest_node = min(vp_nodes, key=lambda x: abs(current_price - x))

    if direction == "LONG":
        # SL slightly below key level, TP above
        sl = nearest_node * 0.997  # little below
        # minimal RR 1:1.9
        risk = current_price - sl
        tp = current_price + 1.9 * risk
    elif direction == "SHORT":
        sl = nearest_node * 1.003  # little above
        risk = sl - current_price
        tp = current_price - 1.9 * risk
    else:
        return None

    if risk <= 0:
        return None

    rr = abs(tp - current_price) / (abs(current_price - sl) + 1e-9)

    if prob < 0.85 or rr < 1.9:
        return None

    return {
        "symbol": symbol,
        "yf_symbol": yf_symbol,
        "direction": direction,
        "prob": prob,
        "entry": round(current_price, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "rr": rr,
    }


# ============================================================
# TELEGRAM HANDLERS
# ============================================================


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /start → start autoscan (5m, every 5 min) + keep manual commands working.
    """
    chat_id = update.effective_chat.id

    # Setup repeating job
    job_queue = context.job_queue
    existing_job = context.chat_data.get("autoscan_job")
    if existing_job:
        existing_job.schedule_removal()

    job = job_queue.run_repeating(
        autoscan_job,
        interval=5 * 60,  # every 5 min
        first=5,  # start after 5 sec
        data={
            "chat_id": chat_id,
            "last_signal_time": None,
        },
    )
    context.chat_data["autoscan_job"] = job

    msg = (
        "🚀 Auto-scan started.\n\n"
        "- Every 5 minutes I scan selected NSE F&O stocks (5m timeframe).\n"
        "- I only send signals when probability ≥ 85% and RR ≥ 1.9.\n"
        "- Logic: fixed-range volume profile + EMA trend + candlestick confluence.\n"
        "- Cooldown ≈ 10 minutes after I send signals.\n\n"
        "Manual analysis still works, e.g. `/dmart` or `/reliance 4h`.\n"
        "Use /stop to stop autoscan."
    )
    await update.message.reply_text(msg)


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /stop → stop autoscan.
    """
    job = context.chat_data.get("autoscan_job")
    if job:
        job.schedule_removal()
        context.chat_data["autoscan_job"] = None
        await update.message.reply_text("🛑 Auto-scan stopped. Manual /stock analysis still works.")
    else:
        await update.message.reply_text("No auto-scan is currently running.")


async def options_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /options → list optionable stocks/indices.
    """
    text = (
        "📜 Optionable NSE underlyings I support (you can extend in code):\n\n"
        + ", ".join(sorted(OPTIONABLE_SYMBOLS))
        + "\n\nExample usage:\n"
        "`/strikeprice NIFTY50 25900`\n"
        "(I will analyse underlying + strike and suggest CE/PE + expiry.)"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def strikeprice_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /strikeprice <symbol> <strike>
    Example: /strikeprice NIFTY50 25900
    """
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "Usage: `/strikeprice <symbol> <strike>`\n"
            "Example: `/strikeprice NIFTY50 25900`",
            parse_mode="Markdown",
        )
        return

    symbol = args[0].upper()
    try:
        strike = float(args[1])
    except ValueError:
        await update.message.reply_text("Strike price must be a number, e.g. 25900")
        return

    yf_symbol = map_user_symbol(symbol)
    if not yf_symbol:
        await update.message.reply_text(f"Unknown symbol: {symbol}")
        return

    df = fetch_ohlcv(yf_symbol, "1d")
    if df is None or df.empty:
        await update.message.reply_text(
            f"Could not fetch data for {symbol} ({yf_symbol})."
        )
        return

    last_price = float(df["close"].iloc[-1])

    await update.message.reply_text(
        f"🔍 Analysing options for {symbol} @ strike {strike:.2f} "
        f"(underlying ≈ {last_price:.2f}) using Gemini...",
    )

    try:
        text = await gemini_analyze_options(symbol, yf_symbol, strike, last_price)
    except Exception as e:
        logger.exception(f"Error in Gemini options analysis: {e}")
        await update.message.reply_text(
            "Gemini options analysis failed. Please try again later."
        )
        return

    await update.message.reply_text(text)


async def autoscan_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    JobQueue callback: scan futures list every 5 min and send signals
    if probability ≥ 85% and RR ≥ 1.9, with 10 min cooldown after sending.
    """
    job_data = context.job.data or {}
    chat_id = job_data.get("chat_id")

    if chat_id is None:
        return

    now = dt.datetime.utcnow()
    last_signal_time = job_data.get("last_signal_time")

    # Cooldown of 10 minutes after last signals
    if last_signal_time and (now - last_signal_time).total_seconds() < 10 * 60:
        return

    signals: List[Dict[str, Any]] = []

    for symbol in FUTURES_SCAN_LIST:
        yf_symbol = map_user_symbol(symbol)
        if not yf_symbol:
            continue

        try:
            sig = build_autoscan_signal(symbol, yf_symbol)
        except Exception as e:
            logger.exception(f"Autoscan error for {symbol}: {e}")
            sig = None

        if sig:
            signals.append(sig)

    if not signals:
        return

    # We got signals → mark cooldown start
    job_data["last_signal_time"] = now
    context.job.data = job_data

    lines = []
    for s in signals:
        lines.append(
            f"📊 *{s['symbol']}* — {s['direction']}\n"
            f"Entry: `{s['entry']}`\n"
            f"SL: `{s['sl']}`\n"
            f"TP: `{s['tp']}`\n"
            f"RR: `{s['rr']:.2f}`  | Prob ≈ `{int(s['prob'] * 100)}%`"
        )

    text = "⚡️ Auto-scan signals (5m timeframe):\n\n" + "\n\n".join(lines)
    await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")


async def generic_stock_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Any other command is treated as a stock symbol:
    - /dmart
    - /dmart 4h
    - /reliance daily
    etc.
    Reserved commands: start, stop, options, strikeprice, help are ignored here.
    """
    if not update.message:
        return

    text = update.message.text.strip()
    parts = text.split()

    if not parts:
        return

    command = parts[0]  # e.g. "/dmart"
    cmd_name = command.lstrip("/").split("@")[0].upper()

    # Skip reserved
    if cmd_name in {"START", "STOP", "OPTIONS", "STRIKEPRICE", "HELP"}:
        return

    timeframe = None
    if len(parts) >= 2:
        timeframe = parts[1].lower()
        # normalise some variations
        if timeframe in {"4hr", "4hour"}:
            timeframe = "4h"
        if timeframe in {"1day", "day"}:
            timeframe = "1d"
        if timeframe in {"week", "1week"}:
            timeframe = "1w"

    yf_symbol = map_user_symbol(cmd_name)
    if not yf_symbol:
        await update.message.reply_text(f"Unknown symbol: {cmd_name}")
        return

    await update.message.reply_text(
        f"📈 Analysing *{cmd_name}* ({yf_symbol}) "
        f"{'on ' + timeframe + ' timeframe ' if timeframe else 'on multi-timeframes '}using Gemini...",
        parse_mode="Markdown",
    )

    multi_tf_data, error_msg = build_multi_tf_summary(cmd_name, yf_symbol, timeframe)
    if not multi_tf_data:
        await update.message.reply_text(
            f"Could not compute indicators for {cmd_name}.\n{error_msg}"
        )
        return

    if error_msg:
        await update.message.reply_text(
            f"⚠️ Some timeframes could not be loaded:\n{error_msg}"
        )

    try:
        analysis_text = await gemini_analyze_stock(
            cmd_name, yf_symbol, timeframe, multi_tf_data
        )
    except Exception as e:
        logger.exception(f"Error in Gemini stock analysis: {e}")
        await update.message.reply_text(
            "Gemini stock analysis failed. Please try again later."
        )
        return

    await update.message.reply_text(analysis_text)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "🧠 *Gemini AI Stock & Options Bot*\n\n"
        "Commands:\n"
        "- `/start` → start 5m autoscan on F&O stocks (signals only when prob ≥ 85% & RR ≥ 1.9).\n"
        "- `/stop` → stop autoscan.\n"
        "- `/options` → list underlyings suitable for options.\n"
        "- `/strikeprice <symbol> <strike>` → options view with CE/PE, expiry, TP/SL.\n"
        "- `/<symbol>` → stock analysis with probabilities.\n"
        "   e.g. `/dmart`, `/reliance`, `/dmart 4h`, `/nifty50 daily`.\n\n"
        "Analysis style:\n"
        "- Elite risk management (top 1% trader mindset).\n"
        "- Uses multi-timeframe OHLCV, EMA, RSI, VWAP & fixed-range volume profile.\n"
        "- If probability ≥ 75% for a move, gives entry, SL (protected level), TP.\n"
        "- If flat/choppy probability is high, will tell you to stay out.\n"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    # Core commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("options", options_cmd))
    application.add_handler(CommandHandler("strikeprice", strikeprice_cmd))

    # Generic stock commands (any other /COMMAND)
    application.add_handler(
        MessageHandler(filters.COMMAND, generic_stock_cmd)
    )

    logger.info("Bot is starting with polling...")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
