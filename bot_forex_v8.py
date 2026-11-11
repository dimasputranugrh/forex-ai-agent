#!/usr/bin/env python3
"""
FOREX & GOLD AI AGENT V8 MASTERPIECE (24/7 Ubuntu VPS Service)
Metode:
1. SMC Liquidity Sweep (Turtle Soup 20-Candle)
2. SMC Fair Value Gap (FVG Institutional Displacement)
3. Intraday Trend (MACD Crossover + EMA 200)
4. Turtle Breakout (Donchian 20-Candle + ADX >= 20)
5. Scalping Pullback (EMA 21 Searah Tren EMA 200)
+ Filter Multi-Timeframe H1 + Pre-Alert 1-5 Menit Sebelumnya
+ Interaktif via Telegram Chat (/status, /analisa, /gold, /btc)
"""

import os
import sys
import time
import math
import json
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

from forex_learning_db import (
    save_signal,
    evaluate_pending_signals,
    get_historical_win_rate,
    get_recap_summary,
    has_active_signal,
    count_active_signals,
)
from forex_ai_brain import evaluate_signal_with_ai

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("forex_agent.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("forex-v8")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL_SEC", "8"))


def get_forex_spot_gold_price() -> float | None:
    """Ambil harga real-time Spot Forex XAU/USD langsung dari Liquidity Provider Bank (identik Exness MT5)."""
    try:
        r = requests.get(
            "https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD",
            timeout=4,
        )
        if r.status_code == 200:
            data = r.json()
            prices = data[0]["spreadProfilePrices"][0]
            return (float(prices["bid"]) + float(prices["ask"])) / 2.0
    except Exception:
        pass
    return None


# ==========================================================
# 1. TELEGRAM ENGINE (KIRIM NOTIF & BALAS CHAT DARI HP)
# ==========================================================
def send_telegram(text: str, chat_id: str = None) -> bool:
    target_chat = chat_id or CHAT_ID
    if not BOT_TOKEN or "GANTI_" in BOT_TOKEN or not target_chat:
        log.warning("Token/Chat ID belum diisi di .env. Pesan: %s", text.splitlines()[0])
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            data={"chat_id": target_chat, "text": text},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        log.error("Gagal kirim Telegram: %s", e)
        return False


# ==========================================================
# 2. DATA ENGINE (XAUUSD & BTCUSD REAL-TIME M5 + H1)
# ==========================================================
def fetch_candles(symbol: str = "GOLD", interval: str = "5m", limit: int = 250) -> pd.DataFrame:
    """
    Mengambil data candlestick real-time M5 / H1.
    - GOLD (XAUUSD): Menggunakan PAXGUSDT / XAUUSD spot feed (bebas limit & buka 24/7)
      dengan fallback Yahoo Finance (GC=F).
    - BTC (BTCUSD): Menggunakan BTCUSDT spot feed.
    """
    # Coba via Binance Spot API (cepat, stabil di VPS, tidak kena blokir, presisi per detik)
    b_sym = "PAXGUSDT" if symbol.upper() == "GOLD" else "BTCUSDT"
    b_int = "5m" if interval == "5m" else "1h"
    urls = [
        f"https://api.binance.com/api/v3/klines?symbol={b_sym}&interval={b_int}&limit={limit}",
        f"https://data-api.binance.vision/api/v3/klines?symbol={b_sym}&interval={b_int}&limit={limit}",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                raw = r.json()
                if isinstance(raw, list) and len(raw) >= 50:
                    df = pd.DataFrame(
                        raw,
                        columns=[
                            "time", "open", "high", "low", "close", "volume",
                            "close_time", "qav", "num_trades", "tbbav", "tbqav", "ignore"
                        ],
                    )
                    df["time"] = pd.to_datetime(df["time"], unit="ms")
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = df[col].astype(float)
                    # Kalibrasi otomatis ke harga Spot Forex Bank (Swissquote/Exness XAUUSDm)
                    if symbol.upper() == "GOLD":
                        spot_xau = get_forex_spot_gold_price()
                        if spot_xau is not None:
                            basis_diff = spot_xau - float(df["close"].iloc[-1])
                            for col in ["open", "high", "low", "close"]:
                                df[col] = df[col] + basis_diff
                    return df[["time", "open", "high", "low", "close", "volume"]]
        except Exception:
            pass

    # Fallback ke Yahoo Finance jika Binance tidak terjangkau
    try:
        import yfinance as yf
        yf_sym = "GC=F" if symbol.upper() == "GOLD" else "BTC-USD"
        yf_int = "5m" if interval == "5m" else "60m"
        period = "5d" if interval == "5m" else "1mo"
        df = yf.Ticker(yf_sym).history(period=period, interval=yf_int)
        if not df.empty:
            df = df.reset_index()
            t_col = "Datetime" if "Datetime" in df.columns else "Date"
            df = df.rename(
                columns={
                    t_col: "time",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            return df[["time", "open", "high", "low", "close", "volume"]].tail(limit).reset_index(drop=True)
    except Exception as e:
        log.error("Gagal ambil data %s (%s): %s", symbol, interval, e)

    return pd.DataFrame()


# ==========================================================
# 3. INDIKATOR TEKNIKAL (EMA, RSI, MACD, ATR, ADX)
# ==========================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMA 21, 50, 200
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50.0)

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd_main"] = ema12 - ema26
    df["macd_sig"] = df["macd_main"].ewm(span=9, adjust=False).mean()

    # ATR 14
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # ADX 14
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_safe = df["atr"].replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_safe
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_safe
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    return df


# ==========================================================
# 4. OTAK ANALISA V8 MASTERPIECE (5 METODE + PRE-ALERT)
# ==========================================================
def analyze_market(symbol: str = "GOLD") -> dict:
    df_m5 = fetch_candles(symbol, "5m", 500)
    df_h1 = fetch_candles(symbol, "1h", 500)

    if len(df_m5) < 60:
        return {"ok": False, "error": "Data M5 belum cukup"}

    df_m5 = compute_indicators(df_m5)
    has_h1 = len(df_h1) >= 50
    if has_h1:
        df_h1 = compute_indicators(df_h1)

    # Index: -1 = candle live (sedang jalan), -2 = candle baru saja tutup (c1), -3 = 2 candle lalu (c2), -4 = 3 candle lalu (c3)
    live = df_m5.iloc[-1]
    c1 = df_m5.iloc[-2]
    c2 = df_m5.iloc[-3]
    c3 = df_m5.iloc[-4]

    vol_atr = float(c1["atr"]) if not math.isnan(c1["atr"]) else 1.5
    live_price = float(live["close"])

    # Swing 5 candle & Likuiditas 20 candle sebelumnya (sebelum c1)
    window_5 = df_m5.iloc[-6:-1]
    swing5_high = float(window_5["high"].max())
    swing5_low = float(window_5["low"].min())

    window_20 = df_m5.iloc[-22:-2]
    prev20_high = float(window_20["high"].max())
    prev20_low = float(window_20["low"].min())

    # Institutional Range Relative Position (0.0 = Lantai Support, 1.0 = Atap Resisten)
    range_span = max(prev20_high - prev20_low, vol_atr * 2.0)
    rel_pos = max(0.0, min(1.0, (live_price - prev20_low) / range_span))

    bull_m5 = (live_price > float(c1["ema200"])) and (float(c1["ema21"]) > float(c1["ema50"]))
    bear_m5 = (live_price < float(c1["ema200"])) and (float(c1["ema21"]) < float(c1["ema50"]))

    # Evaluasi Tren H1 yang Kokoh (Tidak flip-flop hanya karena 1 wick sesaat)
    h1_last = df_h1.iloc[-1] if has_h1 else live
    h1_ema200 = float(h1_last["ema200"])
    h1_ema50 = float(h1_last["ema50"])
    h1_ema21 = float(h1_last["ema21"])

    bull_h1 = (live_price >= h1_ema200 - (vol_atr * 1.5)) and (float(c1["ema21"]) >= float(c1["ema50"]))
    bear_h1 = (live_price <= h1_ema200 + (vol_atr * 1.5)) and (float(c1["ema21"]) <= float(c1["ema50"]))

    # Institutional Smart Money Rule:
    # rel_pos < 0.28: Deep Discount (Lantai Support) -> DILARANG KERAS SELL!
    # rel_pos > 0.72: High Premium (Atap Resisten) -> DILARANG KERAS BUY!
    allow_buy = bull_h1 and (rel_pos <= 0.72)
    allow_sell = bear_h1 and (rel_pos >= 0.28)
    strong_trend = float(c1["adx"]) >= 20.0

    # --- CEK PERINGATAN SIAP-SIAP (HANYA JIKA LOLOS FILTER AREA & TREN) ---
    prep_msg = None
    pair_label = "XAUUSDm (Emas)" if symbol.upper() == "GOLD" else "BTCUSDm (Bitcoin)"

    if allow_buy:
        if bull_m5 and float(live["low"]) <= float(live["ema21"]) + (vol_atr * 0.2) and float(live["low"]) >= float(live["ema50"]):
            prep_msg = (
                f"⚠️ [PERINGATAN SIAP-SIAP BUY - AREA DISKON]\n"
                f"Simbol: {pair_label} (M5)\n"
                f"🏷️ Metode: Pullback EMA 21 Searah Tren Bullish\n"
                f"📍 Harga masuk zona serok diskon ({live_price:.2f}).\n"
                f"⏳ Siapkan MT5 di HP/Laptop 1-5 menit! Tunggu aba-aba EKSEKUSI."
            )
        elif float(live["low"]) <= prev20_low:
            prep_msg = (
                f"⚠️ [PERINGATAN SIAP-SIAP BUY - ZONA LIKUIDITAS LANTAI]\n"
                f"Simbol: {pair_label} (M5)\n"
                f"📍 Uji Lantai Support {live_price:.2f} (Potensi Pantulan Rebound)\n"
                f"⏳ Siapkan MT5 1-5 menit! Menunggu konfirmasi penutupan candle."
            )
    elif allow_sell:
        if bear_m5 and float(live["high"]) >= float(live["ema21"]) - (vol_atr * 0.2) and float(live["high"]) <= float(live["ema50"]):
            prep_msg = (
                f"⚠️ [PERINGATAN SIAP-SIAP SELL - AREA PREMIUM]\n"
                f"Simbol: {pair_label} (M5)\n"
                f"🏷️ Metode: Pullback EMA 21 Searah Tren Bearish\n"
                f"📍 Harga masuk zona pantulan jual ({live_price:.2f}).\n"
                f"⏳ Siapkan MT5 di HP/Laptop 1-5 menit! Tunggu aba-aba EKSEKUSI."
            )
        elif float(live["high"]) >= prev20_high:
            prep_msg = (
                f"⚠️ [PERINGATAN SIAP-SIAP SELL - ZONA LIKUIDITAS ATAP]\n"
                f"Simbol: {pair_label} (M5)\n"
                f"📍 Uji Atap Resisten {live_price:.2f} (Potensi Penolakan/Drop)\n"
                f"⏳ Siapkan MT5 1-5 menit! Menunggu konfirmasi penutupan candle."
            )

    # --- CEK 5 METODE EKSEKUSI RESMI (CANDLE BARU SAJA TUTUP / c1) ---
    c_open = float(c1["open"])
    c_close = float(c1["close"])
    c_high = float(c1["high"])
    c_low = float(c1["low"])
    c_body = abs(c_close - c_open)

    valid_bull = (c_close > c_open) and (c_body >= vol_atr * 0.22)
    valid_bear = (c_close < c_open) and (c_body >= vol_atr * 0.22)

    # 1. SMC Liquidity Sweep
    smc_sweep_buy = (c_low < prev20_low) and (c_close > prev20_low) and valid_bull and (bull_h1 or float(c1["rsi"]) < 38.0)
    smc_sweep_sell = (c_high > prev20_high) and (c_close < prev20_high) and valid_bear and ((not bull_h1) or float(c1["rsi"]) > 62.0)

    # 2. SMC Fair Value Gap (FVG)
    smc_fvg_buy = bull_m5 and (c_low > float(c3["high"])) and (float(c2["close"]) > float(c2["open"])) and ((float(c2["close"]) - float(c2["open"])) > vol_atr * 1.1)
    smc_fvg_sell = bear_m5 and (c_high < float(c3["low"])) and (float(c2["close"]) < float(c2["open"])) and ((float(c2["open"]) - float(c2["close"])) > vol_atr * 1.1)

    # 3. Intraday Trend: MACD + EMA 200
    macd_buy = (c_close > float(c1["ema200"])) and (float(c2["macd_main"]) <= float(c2["macd_sig"])) and (float(c1["macd_main"]) > float(c1["macd_sig"])) and (float(c1["macd_main"]) < 0)
    macd_sell = (c_close < float(c1["ema200"])) and (float(c2["macd_main"]) >= float(c2["macd_sig"])) and (float(c1["macd_main"]) < float(c1["macd_sig"])) and (float(c1["macd_main"]) > 0)

    # 4. Turtle Breakout 20 Candle
    breakout_buy = bull_m5 and strong_trend and (c_close > prev20_high) and (c_body > vol_atr * 0.65) and (float(c1["rsi"]) < 72.0)
    breakout_sell = bear_m5 and strong_trend and (c_close < prev20_low) and (c_body > vol_atr * 0.65) and (float(c1["rsi"]) > 28.0)

    # 5. Scalping Pullback EMA 21 Searah Tren
    scalp_buy = bull_m5 and strong_trend and (c_low <= float(c1["ema21"]) + (vol_atr * 0.1)) and (c_close > float(c1["ema21"])) and valid_bull and (46.0 < float(c1["rsi"]) < 67.0)
    scalp_sell = bear_m5 and strong_trend and (c_high >= float(c1["ema21"]) - (vol_atr * 0.1)) and (c_close < float(c1["ema21"])) and valid_bear and (33.0 < float(c1["rsi"]) < 54.0)

    signal_dir = 0
    kategori = ""
    detail = ""
    durasi = ""

    # STRICT MINIMAL RISIKO & SMART MONEY (DILARANG BUY DI ATAP, DILARANG SELL DI LANTAI)
    if allow_buy:
        if smc_sweep_buy or smc_fvg_buy:
            signal_dir = 1
            kategori = "🏦 [KATEGORI: SMC SMART MONEY / PROP FIRM]"
            detail = "Liquidity Sweep (Sapuan Stop Loss Bawah)" if smc_sweep_buy else "Fair Value Gap (Dorongan Institusi)"
            durasi = "Tahan 15 - 60 Menit"
        elif macd_buy:
            signal_dir = 1
            kategori = "🌊 [KATEGORI: INTRADAY TREND (JUARA BACKTEST)]"
            detail = "MACD Crossover di Area Diskon EMA 200"
            durasi = "Santai / Tahan 30 - 120 Menit"
        elif breakout_buy and rel_pos <= 0.65:
            signal_dir = 1
            kategori = "🚀 [KATEGORI: TURTLE BREAKOUT]"
            detail = "Tembus Harga 20 Candle + Momentum ADX Kuat"
            durasi = "Ikuti Tren (15 - 60 Menit)"
        elif scalp_buy:
            signal_dir = 1
            kategori = "⚡ [KATEGORI: SCALPING PULLBACK SEARAH TREN]"
            detail = "Pantulan EMA 21 Searah Tren EMA 200"
            durasi = "Cepat (5 - 20 Menit)"
    elif allow_sell:
        if smc_sweep_sell or smc_fvg_sell:
            signal_dir = -1
            kategori = "🏦 [KATEGORI: SMC SMART MONEY / PROP FIRM]"
            detail = "Liquidity Sweep (Sapuan Stop Loss Atas)" if smc_sweep_sell else "Fair Value Gap (Dorongan Institusi)"
            durasi = "Tahan 15 - 60 Menit"
        elif macd_sell:
            signal_dir = -1
            kategori = "🌊 [KATEGORI: INTRADAY TREND (JUARA BACKTEST)]"
            detail = "MACD Crossover di Area Diskon EMA 200"
            durasi = "Santai / Tahan 30 - 120 Menit"
        elif breakout_sell and rel_pos >= 0.35:
            signal_dir = -1
            kategori = "🚀 [KATEGORI: TURTLE BREAKOUT]"
            detail = "Tembus Harga 20 Candle + Momentum ADX Kuat"
            durasi = "Ikuti Tren (15 - 60 Menit)"
        elif scalp_sell:
            signal_dir = -1
            kategori = "⚡ [KATEGORI: SCALPING PULLBACK SEARAH TREN]"
            detail = "Pantulan EMA 21 Searah Tren EMA 200"
            durasi = "Cepat (5 - 20 Menit)"

    signal_msg = None
    signal_meta = None

    # Kalkulasi Target Realistis M5 (Scalping/Intraday Cepat Tersentuh)
    if symbol.upper() == "GOLD":
        # Emas: SL terukur 20 - 35 pips ($2.0 - $3.5), TP realistis 30 - 52 pips ($3.0 - $5.2)
        sl_dist = max(2.0, min(vol_atr * 1.3, 3.5))
    else:
        # Bitcoin: SL 1.3x ATR
        sl_dist = vol_atr * 1.3

    risk = sl_dist
    rr_ratio = 1.5
    tp_dist = risk * rr_ratio
    pip_scale = 10.0 if symbol.upper() == "GOLD" else 1.0

    if signal_dir == 1:
        searah_h1 = bull_h1
        bintang = "⭐⭐⭐⭐⭐ (SEARAH TREND BESAR H1 - A+)" if searah_h1 else "⭐⭐⭐⭐ (Valid M5)"
        sl = live_price - sl_dist
        tp = live_price + tp_dist
        signal_meta = {
            "action": "BUY",
            "method": kategori.split(":")[-1].replace("]", "").strip(),
            "price": live_price,
            "tp": tp,
            "sl": sl,
            "rr": rr_ratio,
            "tp_pips": tp_dist * pip_scale,
            "sl_pips": sl_dist * pip_scale,
            "bintang": bintang,
            "kategori": kategori,
            "detail": detail,
            "durasi": durasi,
            "pair_label": pair_label,
            "indicators": {
                "rsi": round(float(live["rsi"]), 1),
                "adx": round(float(c1["adx"]), 1),
                "ema200_trend": "BULL" if live_price > float(live["ema200"]) else "BEAR",
                "h1_trend": "BULL" if bull_h1 else "BEAR",
                "rel_pos": round(rel_pos, 2),
            }
        }
    elif signal_dir == -1:
        searah_h1 = not bull_h1
        bintang = "⭐⭐⭐⭐⭐ (SEARAH TREND BESAR H1 - A+)" if searah_h1 else "⭐⭐⭐⭐ (Valid M5)"
        sl = live_price + sl_dist
        tp = live_price - tp_dist
        signal_meta = {
            "action": "SELL",
            "method": kategori.split(":")[-1].replace("]", "").strip(),
            "price": live_price,
            "tp": tp,
            "sl": sl,
            "rr": rr_ratio,
            "tp_pips": tp_dist * pip_scale,
            "sl_pips": sl_dist * pip_scale,
            "bintang": bintang,
            "kategori": kategori,
            "detail": detail,
            "durasi": durasi,
            "pair_label": pair_label,
            "indicators": {
                "rsi": round(float(live["rsi"]), 1),
                "adx": round(float(c1["adx"]), 1),
                "ema200_trend": "BULL" if live_price > float(live["ema200"]) else "BEAR",
                "h1_trend": "BULL" if bull_h1 else "BEAR",
                "rel_pos": round(rel_pos, 2),
            }
        }

    # Susun teks laporan status pasar kapan pun diminta lewat chat /status
    trend_m5_str = "NAIK (Bullish)" if bull_m5 else ("TURUN (Bearish)" if bear_m5 else "TRANSISI / SIDEWAYS")
    trend_h1_str = "BULLISH (Dominan Naik)" if bull_h1 else "BEARISH (Dominan Turun)"
    saran = (
        f"Fokus cari BUY saat harga koreksi menyentuh EMA 21 ({float(live['ema21']):.2f})"
        if bull_m5
        else (
            f"Fokus cari SELL saat harga naik menyentuh EMA 21 ({float(live['ema21']):.2f})"
            if bear_m5
            else "Pasar sedang transisi/sideways. Tunggu sinyal valid dari bot, jangan paksa entry."
        )
    )

    status_report = (
        f"📊 [LAPORAN LIVE 24 JAM - {pair_label}]\n"
        f"📍 Harga Sekarang: {live_price:.2f}\n"
        f"🧭 Trend M5: {trend_m5_str}\n"
        f"🏛️ Trend Besar H1: {trend_h1_str}\n"
        f"📏 Garis Penting M5:\n"
        f"   • EMA 21 (Biru): {float(live['ema21']):.2f}\n"
        f"   • EMA 50 (Merah): {float(live['ema50']):.2f}\n"
        f"   • EMA 200 (Kuning): {float(live['ema200']):.2f}\n"
        f"🔼 Batas Atas 20 Candle: {prev20_high:.2f}\n"
        f"🔽 Batas Bawah 20 Candle: {prev20_low:.2f}\n"
        f"💪 Tenaga ADX: {float(c1['adx']):.1f} | RSI: {float(live['rsi']):.1f}\n"
        f"💡 Saran AI: {saran}"
    )

    return {
        "ok": True,
        "symbol": symbol,
        "bar_time": str(live["time"]),
        "closed_bar_time": str(c1["time"]),
        "prep_msg": prep_msg,
        "signal_msg": signal_msg,
        "signal_meta": signal_meta,
        "status_report": status_report,
        "live_price": live_price,
        "high_m5": float(live["high"]),
        "low_m5": float(live["low"]),
        "close_m5": float(live["close"]),
    }


# ==========================================================
# 5. LOOP UTAMA 24 JAM + PEMBACA PERINTAH TELEGRAM (/status, /btc, dll)
# ==========================================================
def poll_telegram_commands(last_update_id: int, active_symbol: str) -> tuple[int, str]:
    if not BOT_TOKEN or "GANTI_" in BOT_TOKEN:
        return last_update_id, active_symbol

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    try:
        r = requests.get(url, params={"offset": last_update_id + 1, "timeout": 1}, timeout=5)
        if r.status_code != 200:
            return last_update_id, active_symbol

        data = r.json()
        for upd in data.get("result", []):
            last_update_id = upd["update_id"]
            msg = upd.get("message", {})
            text = (msg.get("text") or "").strip().lower()
            sender_id = str(msg.get("chat", {}).get("id", ""))

            if not text:
                continue

            if text in ["/start", "/help", "menu", "halo", "tes"]:
                reply = (
                    "👑 [FOREX & GOLD AI AGENT V8 - 24 JAM AKTIF]\n\n"
                    "Perintah yang bisa kamu ketik kapan saja dari HP:\n"
                    "• /status atau /analisa -> Lihat kondisi harga, EMA, RSI & saran detik ini\n"
                    "• /recap -> Rekap performa akun demo simulasi 0.01 lot (Untung/Rugi)\n"
                    "• /gold -> Pantau Emas (XAUUSDm)\n"
                    "• /btc -> Pantau Bitcoin (BTCUSDm - Cocok Sabtu/Minggu)\n\n"
                    f"Saat ini bot memantau otomatis: {active_symbol} 24 jam nonstop!"
                )
                send_telegram(reply, sender_id)
            elif text in ["/recap", "recap", "/porto", "porto", "rekap"]:
                rec = get_recap_summary()
                sign = "+" if rec["net_usd"] >= 0 else ""
                rp = rec["net_usd"] * 16000  # Estimasi kurs IDR
                pending_txt = f"{len(rec['pending'])} posisi" if rec['pending'] else "Tidak ada"
                recap_msg = (
                    "📊 [REKAP HASIL TRADING AI - 0.01 LOT]\n\n"
                    f"🎯 Total Selesai: {rec['total']} Order\n"
                    f"✅ Menang (Take Profit): {rec['wins']}\n"
                    f"❌ Kalah (Stop Loss): {rec['losses']}\n"
                    f"📈 Win Rate: {rec['win_rate']}%\n"
                    f"🪙 Akumulasi Pips: {rec['total_pips']:+.1f} Pips\n"
                    f"💵 Net PnL (0.01 Lot): {sign}${rec['net_usd']:.2f} USD (~Rp {rp:,.0f})\n"
                    f"⏳ Posisi Aktif: {pending_txt}\n\n"
                    "Setiap sinyal yang muncul otomatis masuk simulasi beli 0.01 lot dan dihitung untung/ruginya."
                )
                send_telegram(recap_msg, sender_id)
            elif text in ["/status", "/analisa", "status", "analisa", "cek"]:
                res = analyze_market(active_symbol)
                send_telegram(res.get("status_report", "Gagal memuat data."), sender_id)
            elif text in ["/gold", "gold", "emas"]:
                active_symbol = "GOLD"
                res = analyze_market("GOLD")
                send_telegram("✅ Pantauan utama diubah ke EMAS (XAUUSDm)!\n\n" + res.get("status_report", ""), sender_id)
            elif text in ["/btc", "btc", "bitcoin"]:
                active_symbol = "BTC"
                res = analyze_market("BTC")
                send_telegram("✅ Pantauan utama diubah ke BITCOIN (BTCUSDm)!\n\n" + res.get("status_report", ""), sender_id)
    except Exception:
        pass

    return last_update_id, active_symbol


def setup_telegram_bot_menu():
    """Daftarkan tombol menu garis tiga (Bot Commands) ke API Telegram."""
    if not BOT_TOKEN or "GANTI_" in BOT_TOKEN:
        return
    commands = [
        {"command": "status", "description": "Analisa teknikal harga & sinyal detik ini"},
        {"command": "recap", "description": "Rekap hasil trading & PnL 0.01 lot"},
        {"command": "gold", "description": "Ubah pantauan utama ke Emas (XAU/USD)"},
        {"command": "btc", "description": "Ubah pantauan utama ke Bitcoin (BTC/USD)"},
        {"command": "help", "description": "Panduan & bantuan perintah bot"},
    ]
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/setMyCommands", json={"commands": commands}, timeout=5)
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/setChatMenuButton", json={"menu_button": {"type": "commands"}}, timeout=5)
    except Exception:
        pass


def main():
    setup_telegram_bot_menu()
    active_symbol = os.getenv("DEFAULT_MARKET", "GOLD").upper()
    log.info("Memulai Forex & Gold AI Agent V8 (Pantauan: %s)...", active_symbol)

    res = analyze_market(active_symbol)
    if res.get("ok"):
        send_telegram(
            "🚀 [SERVER VPS 24 JAM AKTIF - FOREX AI AGENT V8]\n"
            "Laptop kamu sekarang boleh dimatikan! Bot berjalan 24 jam penuh di background.\n"
            "Ketik /status atau /analisa kapan saja di chat ini untuk cek pasar.\n\n"
            + res["status_report"]
        )

    last_signal_bar = None
    last_prep_bar = None
    last_update_id = 0

    while True:
        try:
            # 1. Cek apakah kamu kirim chat (/status, /gold, /btc) dari HP
            last_update_id, active_symbol = poll_telegram_commands(last_update_id, active_symbol)

            # 2. Analisa pasar otomatis
            res = analyze_market(active_symbol)
            if res.get("ok"):
                cur_bar = res["bar_time"]
                closed_bar = res["closed_bar_time"]

                # Evaluasi Sinyal Pending (Self-Learning Loop)
                resolved = evaluate_pending_signals(
                    current_high=res["high_m5"],
                    current_low=res["low_m5"],
                    current_close=res["close_m5"],
                    symbol=active_symbol
                )
                for r in resolved:
                    is_cuan = r["outcome"] == "HIT_TP"
                    icon = "🎉 [TARGET TERCAPAI - TAKE PROFIT!]" if is_cuan else "🛑 [STOP LOSS TERPICU - PROTEKSI MODAL]"
                    pnl_txt = f"+{r['pips']:.1f} Pips" if r['pips'] >= 0 else f"{r['pips']:.1f} Pips"
                    usd_val = r['pips'] * 0.10
                    usd_txt = f"+${usd_val:.2f}" if usd_val >= 0 else f"-${abs(usd_val):.2f}"
                    rec = get_recap_summary()
                    notif_eval = (
                        f"{icon}\n"
                        f"Simbol: {active_symbol} ({r['action']})\n"
                        f"Metode: {r['method']}\n"
                        f"Entry: {r['entry']:.2f} -> Exit: {r['exit']:.2f}\n"
                        f"Hasil: {pnl_txt}\n"
                        f"💵 Hasil Posisi (0.01 Lot): {usd_txt} USD\n"
                        f"📊 Rekap Portofolio: {rec['wins']}W - {rec['losses']}L (WR {rec['win_rate']}%) | Total PnL: {'+' if rec['net_usd']>=0 else ''}${rec['net_usd']:.2f} USD\n"
                        f"🧠 *AI Catatan:* Hasil disimpan ke memori pembelajaran."
                    )
                    send_telegram(notif_eval)
                    log.info("Outcome sinyal #%d resolved: %s (%s)", r["id"], r["outcome"], pnl_txt)

                active_count = count_active_signals(active_symbol)

                # Kirim peringatan Siap-Siap maksimal 1x per candle (jika belum penuh 3 posisi)
                if active_count < 3 and res["prep_msg"] and cur_bar != last_prep_bar:
                    send_telegram(res["prep_msg"])
                    last_prep_bar = cur_bar
                    log.info("Pre-Alert dikirim pada bar %s (Posisi aktif: %d/3)", cur_bar, active_count)

                # Kirim Sinyal Eksekusi BUY/SELL (Maksimal 3 posisi simultan, masing-masing SL/TP mandiri)
                meta = res.get("signal_meta")
                if active_count < 3 and meta and closed_bar != last_signal_bar:
                    last_signal_bar = closed_bar
                    # Validasi AI Brain
                    ai_eval = evaluate_signal_with_ai(
                        symbol=active_symbol,
                        action=meta["action"],
                        method=meta["method"],
                        price=meta["price"],
                        tp=meta["tp"],
                        sl=meta["sl"],
                        rr=meta["rr"],
                        indicators=meta["indicators"]
                    )
                    if not ai_eval.get("approved"):
                        log.info("Sinyal %s %s DITOLAK oleh AI Brain: %s", meta["action"], active_symbol, ai_eval.get("reason"))
                    else:
                        # Susun pesan sinyal dengan kecerdasan AI
                        act_icon = "🟢 [PERINTAH: KLIK BUY SEKARANG!]" if meta["action"] == "BUY" else "🔴 [PERINTAH: KLIK SELL SEKARANG!]"
                        act_label = "Harga Beli" if meta["action"] == "BUY" else "Harga Jual"
                        final_msg = (
                            f"{act_icon} {meta['pair_label']} (M5)\n"
                            f"{meta['kategori']}\n"
                            f"🌟 Kualitas: {meta['bintang']}\n"
                            f"🧠 Teknik: {meta['detail']}\n"
                            f"⏱️ Gaya: {meta['durasi']}\n"
                            f"📈 {act_label}: {meta['price']:.2f}\n"
                            f"🛑 Stop Loss (SL): {meta['sl']:.2f} (-{meta.get('sl_pips', 30):.0f} Pips)\n"
                            f"🎯 Take Profit (TP): {meta['tp']:.2f} (+{meta.get('tp_pips', 45):.0f} Pips | R:R 1:{meta.get('rr', 1.5):.1f})\n"
                            f"💼 Auto-Simulasi Portofolio: Entry 0.01 Lot\n\n"
                            f"🤖 *AI Brain Verdict:* {ai_eval.get('verdict')} (Score: {ai_eval.get('score')}/100)\n"
                            f"💡 *Penalaran AI:* {ai_eval.get('reason')}"
                        )
                        send_telegram(final_msg)
                        save_signal(
                            symbol=active_symbol,
                            action=meta["action"],
                            method=meta["method"],
                            entry=meta["price"],
                            tp=meta["tp"],
                            sl=meta["sl"],
                            rr=meta["rr"],
                            adx=meta["indicators"].get("adx", 0.0),
                            rsi=meta["indicators"].get("rsi", 0.0),
                            ai_score=ai_eval.get("score", 80),
                            ai_verdict=ai_eval.get("verdict", "PASS"),
                            ai_reasoning=ai_eval.get("reason", "")
                        )
                        log.info("Sinyal Eksekusi disetujui AI Brain dan dikirim pada bar %s", closed_bar)

        except Exception as e:
            log.error("Error pada loop utama: %s", e)

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    if "--test" in sys.argv:
        out = analyze_market("GOLD")
        assert out["ok"], f"Test gagal: {out}"
        print(out["status_report"])
        sys.exit(0)
    main()
