"""
SQLite Transaction Database & Audit Repository for SecureBank AI.
Maintains persistent transaction logs, review queues, and alert history.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_FILE = "data/fraud_intelligence.db"

def get_db_connection(db_path: str = DB_FILE) -> sqlite3.Connection:
    """Creates directory and returns SQLite connection."""
    dir_name = os.path.dirname(db_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str = DB_FILE):
    """Initializes tables for transactions, alerts, and audit logs."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # 1. Transactions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT UNIQUE NOT NULL,
        created_at TEXT NOT NULL,
        amount REAL NOT NULL,
        hour_elapsed INTEGER,
        fraud_probability REAL NOT NULL,
        risk_score REAL NOT NULL,
        risk_level TEXT NOT NULL,
        prediction TEXT NOT NULL,
        anomaly_flag INTEGER NOT NULL,
        action_code TEXT NOT NULL,
        review_status TEXT DEFAULT 'Pending',
        analyst_notes TEXT DEFAULT '',
        features_json TEXT
    );
    """)
    
    # 2. Alerts Queue Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fraud_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        rule_id TEXT NOT NULL,
        severity TEXT NOT NULL,
        message TEXT NOT NULL,
        status TEXT DEFAULT 'NEW',
        FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id)
    );
    """)
    
    conn.commit()
    conn.close()

def log_transaction(
    transaction_id: str,
    amount: float,
    fraud_probability: float,
    risk_score: float,
    risk_level: str,
    prediction: str,
    anomaly_flag: int,
    action_code: str,
    hour_elapsed: int = 0,
    features_json: str = "{}",
    alerts: Optional[List[Dict[str, Any]]] = None,
    db_path: str = DB_FILE
) -> bool:
    """Persists an evaluated transaction and dispatches any alerts."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        cursor.execute("""
        INSERT OR REPLACE INTO transactions (
            transaction_id, created_at, amount, hour_elapsed,
            fraud_probability, risk_score, risk_level, prediction,
            anomaly_flag, action_code, review_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            transaction_id, now_str, amount, hour_elapsed,
            fraud_probability, risk_score, risk_level, prediction,
            anomaly_flag, action_code, 'Pending'
        ))
        
        # Log individual alerts
        if alerts:
            for alert in alerts:
                if alert.get("rule_id") != "R000":  # Skip INFO/No alert
                    cursor.execute("""
                    INSERT INTO fraud_alerts (
                        transaction_id, created_at, rule_id, severity, message, status
                    ) VALUES (?, ?, ?, ?, ?, 'NEW')
                    """, (
                        transaction_id, now_str,
                        alert.get("rule_id", "GEN"),
                        alert.get("severity", "MEDIUM"),
                        alert.get("message", "Suspicious transaction pattern")
                    ))
                    
        conn.commit()
        return True
    except Exception as e:
        print(f"Database logging error: {e}")
        return False
    finally:
        conn.close()

def get_recent_transactions(limit: int = 50, db_path: str = DB_FILE) -> pd.DataFrame:
    """Returns recent transactions DataFrame."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    query = "SELECT * FROM transactions ORDER BY id DESC LIMIT ?"
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df

def get_active_alerts(db_path: str = DB_FILE) -> pd.DataFrame:
    """Returns all active fraud alerts joined with transaction data."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    query = """
    SELECT 
        a.id as alert_id, a.transaction_id, a.created_at, a.rule_id, a.severity, a.message, a.status,
        t.amount, t.risk_score, t.risk_level, t.prediction
    FROM fraud_alerts a
    JOIN transactions t ON a.transaction_id = t.transaction_id
    ORDER BY a.id DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def update_transaction_status(
    transaction_id: str,
    new_status: str,
    analyst_notes: str = "",
    db_path: str = DB_FILE
) -> bool:
    """Updates review queue status: 'Approved', 'Confirmed Fraud', 'False Positive'."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("""
        UPDATE transactions 
        SET review_status = ?, analyst_notes = ?
        WHERE transaction_id = ?
        """, (new_status, analyst_notes, transaction_id))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error updating transaction status: {e}")
        return False
    finally:
        conn.close()

def get_database_kpis(db_path: str = DB_FILE) -> Dict[str, Any]:
    """Calculates operational metrics across all logged transactions."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM transactions")
    total_tx = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE prediction = 'Fraud'")
    fraud_tx = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM transactions WHERE review_status = 'Pending' AND risk_score >= 60")
    urgent_reviews = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COALESCE(SUM(amount), 0.0) FROM transactions WHERE prediction = 'Fraud'")
    prevented_loss = cursor.fetchone()[0] or 0.0
    
    conn.close()
    return {
        "total_transactions": total_tx,
        "fraud_transactions": fraud_tx,
        "fraud_rate_pct": (fraud_tx / total_tx * 100.0) if total_tx > 0 else 0.0,
        "urgent_reviews_pending": urgent_reviews,
        "prevented_loss_usd": round(prevented_loss, 2)
    }

# Initialize database tables on import
init_db()
