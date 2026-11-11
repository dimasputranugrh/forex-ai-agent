"""
Database Memori & Self-Learning Engine untuk Forex & Gold AI Agent
Merekam sinyal, mengevaluasi win/loss real-time, mendeteksi jam/metode terbaik,
dan memblokir setup berisiko tinggi berdasarkan riwayat empiris.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parent / "forex_memory.db"
log = logging.getLogger("forex_memory")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db():
    conn = get_db_connection()
    with conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS signal_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,          -- BUY / SELL
            method TEXT NOT NULL,          -- SMC_SWEEP, SMC_FVG, MACD_TREND, TURTLE, SCALP_EMA
            entry_price REAL NOT NULL,
            take_profit REAL NOT NULL,
            stop_loss REAL NOT NULL,
            risk_reward REAL NOT NULL,
            hour_utc INTEGER NOT NULL,
            timeframe TEXT NOT NULL,
            adx REAL,
            rsi REAL,
            atr REAL,
            ai_score INTEGER,
            ai_verdict TEXT,
            ai_reasoning TEXT,
            -- Evaluasi outcome
            status TEXT DEFAULT 'PENDING',  -- PENDING, HIT_TP, HIT_SL, EXPIRED
            exit_price REAL,
            exit_time TEXT,
            pnl_pips REAL DEFAULT 0,
            outcome_notes TEXT
        );

        CREATE TABLE IF NOT EXISTS learned_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_key TEXT UNIQUE NOT NULL,
            description TEXT NOT NULL,
            win_count INTEGER DEFAULT 0,
            loss_count INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0.0,
            is_active BOOLEAN DEFAULT 1,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS market_regime_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            regime TEXT NOT NULL,          -- TRENDING_BULL, TRENDING_BEAR, CHOPPY_RANGING, HIGH_VOLATILITY
            detected_at TEXT NOT NULL,
            recommended_method TEXT
        );
        """)
    conn.close()


def save_signal(
    symbol: str,
    action: str,
    method: str,
    entry: float,
    tp: float,
    sl: float,
    rr: float,
    adx: float = 0.0,
    rsi: float = 0.0,
    atr: float = 0.0,
    ai_score: int = 80,
    ai_verdict: str = "PASS",
    ai_reasoning: str = ""
) -> int:
    init_db()
    conn = get_db_connection()
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    ts = now_utc.isoformat()

    with conn:
        cur = conn.execute("""
            INSERT INTO signal_history (
                timestamp, symbol, action, method, entry_price, take_profit, stop_loss,
                risk_reward, hour_utc, timeframe, adx, rsi, atr, ai_score, ai_verdict, ai_reasoning
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'M5', ?, ?, ?, ?, ?, ?)
        """, (ts, symbol.upper(), action.upper(), method, entry, tp, sl, rr, hour, adx, rsi, atr, ai_score, ai_verdict, ai_reasoning))
        sig_id = cur.lastrowid
    conn.close()
    return sig_id


def evaluate_pending_signals(current_high: float, current_low: float, current_close: float, symbol: str = "GOLD") -> list[dict]:
    """Cek apakah sinyal yang pending sudah kena TP atau SL secara objektif."""
    init_db()
    conn = get_db_connection()
    now_str = datetime.now(timezone.utc).isoformat()
    resolved = []

    with conn:
        rows = conn.execute("""
            SELECT id, action, method, entry_price, take_profit, stop_loss, hour_utc
            FROM signal_history
            WHERE status = 'PENDING' AND symbol = ?
            ORDER BY id ASC
        """, (symbol.upper(),)).fetchall()

        for r in rows:
            sid = r["id"]
            action = r["action"]
            entry = r["entry_price"]
            tp = r["take_profit"]
            sl = r["stop_loss"]
            method = r["method"]

            hit_tp = False
            hit_sl = False
            exit_px = current_close

            if action == "BUY":
                if current_high >= tp:
                    hit_tp = True
                    exit_px = tp
                elif current_low <= sl:
                    hit_sl = True
                    exit_px = sl
            elif action == "SELL":
                if current_low <= tp:
                    hit_tp = True
                    exit_px = tp
                elif current_high >= sl:
                    hit_sl = True
                    exit_px = sl

            if hit_tp or hit_sl:
                outcome = "HIT_TP" if hit_tp else "HIT_SL"
                # Pip kalkulasi untuk Gold ($0.10 = 1 pip)
                pips = ((exit_px - entry) if action == "BUY" else (entry - exit_px)) * 10.0
                conn.execute("""
                    UPDATE signal_history
                    SET status = ?, exit_price = ?, exit_time = ?, pnl_pips = ?,
                        outcome_notes = ?
                    WHERE id = ?
                """, (outcome, exit_px, now_str, pips, f"Evaluasi otomatis candle: {'Cuan Target TP' if hit_tp else 'Kena Proteksi SL'}", sid))

                # Update learned rules
                _update_learning_stats(conn, method, r["hour_utc"], hit_tp)
                resolved.append({
                    "id": sid,
                    "action": action,
                    "method": method,
                    "outcome": outcome,
                    "pips": pips,
                    "entry": entry,
                    "exit": exit_px
                })

    conn.close()
    return resolved


def _update_learning_stats(conn: sqlite3.Connection, method: str, hour_utc: int, is_win: bool):
    rule_keys = [f"METHOD_{method}", f"HOUR_{hour_utc}"]
    now_str = datetime.now(timezone.utc).isoformat()

    for rk in rule_keys:
        row = conn.execute("SELECT win_count, loss_count FROM learned_rules WHERE rule_key = ?", (rk,)).fetchone()
        if row:
            w = row["win_count"] + (1 if is_win else 0)
            l = row["loss_count"] + (0 if is_win else 1)
            wr = (w / (w + l)) * 100.0 if (w + l) > 0 else 0.0
            conn.execute("""
                UPDATE learned_rules
                SET win_count = ?, loss_count = ?, win_rate = ?, updated_at = ?
                WHERE rule_key = ?
            """, (w, l, wr, now_str, rk))
        else:
            w = 1 if is_win else 0
            l = 0 if is_win else 1
            wr = 100.0 if is_win else 0.0
            desc = f"Aturan empiris untuk {rk}"
            conn.execute("""
                INSERT INTO learned_rules (rule_key, description, win_count, loss_count, win_rate, is_active, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, ?)
            """, (rk, desc, w, l, wr, now_str))


def get_historical_win_rate(method: str, hour_utc: int) -> dict:
    """Ambil histori performa agar AI bisa menolak setup yang terbukti sering boncos."""
    init_db()
    conn = get_db_connection()
    res = {"method_win_rate": None, "hour_win_rate": None, "total_trades": 0, "overall_win_rate": 0.0}

    row_total = conn.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN status = 'HIT_TP' THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN status = 'HIT_SL' THEN 1 ELSE 0 END) as losses
        FROM signal_history
        WHERE status IN ('HIT_TP', 'HIT_SL')
    """).fetchone()

    if row_total and row_total["total"] > 0:
        total = row_total["total"]
        wins = row_total["wins"] or 0
        res["total_trades"] = total
        res["overall_win_rate"] = round((wins / total) * 100.0, 1)

    r_method = conn.execute("SELECT win_count, loss_count, win_rate FROM learned_rules WHERE rule_key = ?", (f"METHOD_{method}",)).fetchone()
    if r_method and (r_method["win_count"] + r_method["loss_count"] >= 15):
        res["method_win_rate"] = round(r_method["win_rate"], 1)

    r_hour = conn.execute("SELECT win_count, loss_count, win_rate FROM learned_rules WHERE rule_key = ?", (f"HOUR_{hour_utc}",)).fetchone()
    if r_hour and (r_hour["win_count"] + r_hour["loss_count"] >= 15):
        res["hour_win_rate"] = round(r_hour["win_rate"], 1)

    conn.close()
    return res


def count_active_signals(symbol: str = "GOLD") -> int:
    """Hitung jumlah order PENDING aktif saat ini."""
    init_db()
    conn = get_db_connection()
    row = conn.execute("SELECT COUNT(*) as count FROM signal_history WHERE status = 'PENDING' AND symbol = ?", (symbol.upper(),)).fetchone()
    cnt = row["count"] if row else 0
    conn.close()
    return cnt


def has_active_signal(symbol: str = "GOLD", max_slots: int = 3) -> bool:
    """True jika kapasitas posisi (maksimal 3 slot) sudah penuh."""
    return count_active_signals(symbol) >= max_slots


def get_recap_summary() -> dict:
    """Rekap akumulasi performa akun demo simulasi 0.01 lot."""
    init_db()
    conn = get_db_connection()
    row = conn.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN status = 'HIT_TP' THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN status = 'HIT_SL' THEN 1 ELSE 0 END) as losses,
               SUM(pnl_pips) as total_pips
        FROM signal_history
        WHERE status IN ('HIT_TP', 'HIT_SL')
    """).fetchone()

    total = row["total"] or 0
    wins = row["wins"] or 0
    losses = row["losses"] or 0
    total_pips = row["total_pips"] or 0.0
    win_rate = (wins / total * 100.0) if total > 0 else 0.0
    net_usd = total_pips * 0.10

    pending_rows = conn.execute("""
        SELECT id, symbol, action, method, entry_price, take_profit, stop_loss, timestamp
        FROM signal_history
        WHERE status = 'PENDING'
        ORDER BY id DESC
    """).fetchall()

    conn.close()
    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "total_pips": round(total_pips, 1),
        "net_usd": round(net_usd, 2),
        "pending": [dict(p) for p in pending_rows]
    }
