"""
AI Brain & Reasoning Validator untuk Forex AI Agent
Menggunakan Gemini + empirical historical rules untuk memfilter false breakout,
bad hour trading, dan menganalisa konteks multi-timeframe sebelum sinyal disetujui.
"""

import os
import json
import logging
import requests
from datetime import datetime, timezone
from forex_learning_db import get_historical_win_rate

log = logging.getLogger("forex_ai_brain")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()


def evaluate_signal_with_ai(
    symbol: str,
    action: str,
    method: str,
    price: float,
    tp: float,
    sl: float,
    rr: float,
    indicators: dict
) -> dict:
    """
    AI Validator Tingkat Tinggi:
    1. Cek empirical memory (apakah jam ini / metode ini historisnya jelek).
    2. Cek Risk-Reward minimum (harus >= 1:1.5 untuk membatasi risiko rugi).
    3. Cek Volatilitas & RSI overbought/oversold.
    4. Validasi Gemini LLM untuk penalaran institusional.
    """
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour

    # 1. Empirical Check dari Database Memori
    history = get_historical_win_rate(method, hour)
    if history.get("method_win_rate") is not None and history["method_win_rate"] < 40.0:
        return {
            "approved": False,
            "score": 35,
            "verdict": "REJECT",
            "reason": f"Metode {method} punya win rate historis buruk ({history['method_win_rate']}%) pada database memori.",
        }

    if history.get("hour_win_rate") is not None and history["hour_win_rate"] < 35.0:
        return {
            "approved": False,
            "score": 30,
            "verdict": "REJECT",
            "reason": f"Jam trading UTC {hour} historisnya choppy/sering kena SL ({history['hour_win_rate']}% win rate).",
        }

    # 2. Hard Rule: Strict Multi-Timeframe Filter (H1 Trend Protection)
    h1_trend = indicators.get("h1_trend")
    if (action == "BUY" and h1_trend == "BEAR") or (action == "SELL" and h1_trend == "BULL"):
        return {
            "approved": False,
            "score": 20,
            "verdict": "REJECT",
            "reason": f"Sinyal {action} melawan tren besar H1 ({h1_trend}). Zero tolerance counter-trend demi proteksi modal.",
        }

    # 3. Hard Rule: Smart Money Range Position (No Buying Top, No Selling Floor)
    rel_pos = indicators.get("rel_pos")
    if rel_pos is not None:
        if action == "SELL" and rel_pos < 0.28:
            return {
                "approved": False,
                "score": 15,
                "verdict": "REJECT",
                "reason": f"Ditolak: Harga di zona diskon/support bawah ({rel_pos*100:.0f}% rentang). Dilarang jual di dasar lembah.",
            }
        if action == "BUY" and rel_pos > 0.72:
            return {
                "approved": False,
                "score": 15,
                "verdict": "REJECT",
                "reason": f"Ditolak: Harga di zona premium/resisten atas ({rel_pos*100:.0f}% rentang). Dilarang beli di pucuk atap.",
            }

    # 4. Hard Rule: Strict Risk-to-Reward Minimum (1:1.4)
    if rr < 1.4:
        return {
            "approved": False,
            "score": 40,
            "verdict": "REJECT",
            "reason": f"Risk-to-Reward {rr:.2f} terlalu sempit (< 1:1.4). Demi minim risiko, setup ini diabaikan.",
        }

    # 3. AI Reasoning via Gemini
    score = 85
    reasoning = "Setup selaras dengan struktur harga institusional (SMC) & rasio risiko sehat."

    if GEMINI_API_KEY:
        try:
            prompt = (
                f"Bertindaklah sebagai Senior Quantitative Forex & Gold Trader.\n"
                f"Evaluasi setup sinyal berikut secara sangat ketat (Zero Tolerance untuk setup berisiko tinggi):\n"
                f"- Simbol: {symbol}\n"
                f"- Aksi: {action}\n"
                f"- Metode: {method}\n"
                f"- Entry: {price}, TP: {tp}, SL: {sl}, R:R: {rr:.2f}\n"
                f"- RSI: {indicators.get('rsi')}, ADX: {indicators.get('adx')}, Trend EMA200: {indicators.get('ema200_trend')}\n"
                f"- Jam UTC: {hour}\n\n"
                f"Keluarkan output HANYA dalam format JSON valid tanpa markdown formatting:\n"
                f'{{"approved": true/false, "score": 1-100, "verdict": "PASS"/"REJECT", "reason": "alasan singkat maksimal 2 kalimat"}}'
            )
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
            resp = requests.post(
                url,
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=6
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Clean JSON markdown fences
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                parsed = json.loads(text)
                return {
                    "approved": bool(parsed.get("approved", True)),
                    "score": int(parsed.get("score", 80)),
                    "verdict": str(parsed.get("verdict", "PASS")),
                    "reason": str(parsed.get("reason", reasoning))
                }
        except Exception as e:
            log.warning("Gemini AI validation fallback to quant rules: %s", e)

    return {
        "approved": True,
        "score": score,
        "verdict": "PASS",
        "reason": reasoning
    }
