# ============================================================
# app.py — Indian Stocks AI Bot using OpenRouter (Llama 3.1 70B)
# No quota problems, fast, stable, webhook
# ============================================================

import os
import logging
import asyncio
import datetime as dt
from typing import Optional, Dict, Any, List

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
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # your Render URL
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stocks-openrouter")

# ============================================================
# SYMBOL MAPS
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
}

OPTIONABLE_SYMBOLS = [
    "NIFTY50", "BANKNIFTY", "FINNIFTY",
    "RELIANCE", "HDFCBANK", "ICICIBANK",
    "SBIN", "TCS", "INFY", "AXISBANK",
    "KOTAKBANK", "DMART",
]

FUTURES_SCAN_LIST = [
    "RELIANCE", "HDFCBANK", "ICICIBANK",
    "SBIN", "AXISBANK", "KOTAKBANK",
    "TCS", "INFY", "DMART",
    "NIFTY50", "BANKNIFTY",
]

# ============================================================
# TIMEFRAME MAP
# ============================================================

TIMEFRAME_YF_MAP = {
    "15m": ("15m", "20d"),
    "1h": ("60m", "60d"),
    "4h": ("60m", "60d"),
    "1d": ("1d", "2y"),
}

DEFAULT_MULTI_TF = ["15m", "1h", "4h", "1d"]

# ============================================================
# UTILITIES
# ============================================================

def map_symbol(s: str) -> str:
    s = s.strip().upper()
    return STOCK_SYMBOL_MAP.get(s, f"{s}.NS")

# ============================================================
# FETCH OHLCV (WITH MULTIINDEX FIX)
# ============================================================

def fetch_ohlcv(symbol, timeframe):
    if timeframe not in TIMEFRAME_YF_MAP:
        return None

    interval, period = TIMEFRAME_YF_MAP[timeframe]

    df = yf.download(symbol, interval=interval, period=period, progress=False)
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }).dropna()

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["volume"])

    return df


# ============================================================
# INDICATORS (RSI FIX)
# ============================================================

def add_indicators(df):
    df = df.copy()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["ema_200"] = df["close"].ewm(span=200).mean()

    delta = df["close"].diff().to_numpy().ravel()
    gain = np.where(delta > 0, delta, 0).ravel()
    loss = np.where(delta < 0, -delta, 0).ravel()

    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi"] = (100 - (100 / (1 + rs))).values

    return df


# ============================================================
# OPENROUTER MODEL CALL
# ============================================================

async def openrouter_chat(prompt):
    try:
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model="meta-llama/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=650,
            temperature=0.3,
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenRouter Error: {e}")
        return "OpenRouter request failed."


# ============================================================
# STOCK ANALYSIS
# ============================================================

async def stock_ai(symbol, yf_symbol, timeframe, info):

    prompt = f"""
Act as a top Indian stock analyst.

Symbol: {symbol} ({yf_symbol})
Timeframe: {timeframe}

Data:
Price = {info['price']}
EMA50 = {info['ema_50']}
EMA200 = {info['ema_200']}
RSI = {info['rsi']}

Give:
- Upside probability %
- Downside probability %
- Flat probability %
- Explanation (2 lines)
If highest >= 75%:
- Entry, SL, TP1, TP2, RR

Keep it crisp & powerful.
"""

    return await openrouter_chat(prompt)


# ============================================================
# OPTIONS ANALYSIS
# ============================================================

async def options_ai(symbol, yf_symbol, strike, price):

    prompt = f"""
Act as an NSE Options Expert.

Underlying: {symbol} ({yf_symbol})
LTP: {price}
Requested Strike: {strike}

Give:
• Upside probability %
• Downside probability %
• Flat probability %
• CE/PE Buy/Sell suggestion
• Ideal expiry
• Entry, SL, TP
• RR
• 2–3 lines reasoning
"""

    return await openrouter_chat(prompt)


# ============================================================
# COMMAND HANDLERS
# ============================================================

async def cmd_stock(update: Update, context):
    msg = update.message.text.split()
    symbol = msg[0][1:].upper()
    timeframe = msg[1].lower() if len(msg) > 1 else "1h"

    yf_symbol = map_symbol(symbol)

    await update.message.reply_text("📊 Analysing…")

    df = fetch_ohlcv(yf_symbol, timeframe)
    if df is None:
        return await update.message.reply_text("Data unavailable.")

    df = add_indicators(df)
    last = df.iloc[-1]

    info = {
        "price": float(last["close"]),
        "ema_50": float(last["ema_50"]),
        "ema_200": float(last["ema_200"]),
        "rsi": float(last["rsi"]),
    }

    resp = await stock_ai(symbol, yf_symbol, timeframe, info)
    await update.message.reply_text(resp)


async def cmd_strikeprice(update: Update, context):
    if len(context.args) < 2:
        return await update.message.reply_text("Usage: /strikeprice SYMBOL STRIKE")

    symbol = context.args[0].upper()
    strike = float(context.args[1])

    yf_symbol = map_symbol(symbol)
    df = fetch_ohlcv(yf_symbol, "1d")

    if df is None:
        return await update.message.reply_text("Data unavailable.")

    price = float(df["close"].iloc[-1])

    await update.message.reply_text("📉 Analysing options…")

    resp = await options_ai(symbol, yf_symbol, strike, price)
    await update.message.reply_text(resp)


async def cmd_options(update: Update, context):
    await update.message.reply_text("Optionable NSE symbols:\n" + ", ".join(OPTIONABLE_SYMBOLS))


async def cmd_help(update: Update, context):
    await update.message.reply_text(
        "/dmart\n"
        "/dmart 1h\n"
        "/strikeprice NIFTY50 25900\n"
        "/options\n"
    )


async def error_handler(update, context):
    logger.error(context.error)


# ============================================================
# MAIN WEBHOOK
# ============================================================

def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("options", cmd_options))
    app.add_handler(CommandHandler("strikeprice", cmd_strikeprice))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(MessageHandler(filters.COMMAND, cmd_stock))

    app.add_error_handler(error_handler)

    app.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=TELEGRAM_BOT_TOKEN,
        webhook_url=f"{WEBHOOK_URL}/{TELEGRAM_BOT_TOKEN}",
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()

