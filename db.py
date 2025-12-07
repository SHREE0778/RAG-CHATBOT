"""
Database operations module
"""

import sqlite3
from typing import List, Dict, Optional
from contextlib import contextmanager
import logging
import config

logger = logging.getLogger(__name__)

@contextmanager
def get_connection():
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def init_database():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE)''')
        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id)''')
        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at DESC)''')
        conn.commit()
        logger.info("Database initialized")

def get_or_create_user(username: str) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        if result:
            user_id = result["id"]
            cursor.execute('UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE id = ?', (user_id,))
        else:
            cursor.execute('INSERT INTO users (username) VALUES (?)', (username,))
            user_id = cursor.lastrowid
        conn.commit()
        return user_id

def save_message(user_id: int, role: str, content: str) -> int:
    if role not in ('user', 'assistant', 'system'):
        raise ValueError(f"Invalid role: {role}")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)',
                      (user_id, role, content))
        message_id = cursor.lastrowid
        conn.commit()
        return message_id

def get_chat_history(user_id: int, limit: Optional[int] = None) -> List[Dict]:
    if limit is None:
        limit = config.CHAT_HISTORY_LIMIT
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''SELECT id, role, content, created_at FROM messages 
            WHERE user_id = ? ORDER BY created_at DESC LIMIT ?''', (user_id, limit))
        messages = [{"id": row["id"], "role": row["role"], "content": row["content"],
                    "timestamp": row["created_at"]} for row in reversed(cursor.fetchall())]
        return messages

def clear_chat_history(user_id: int) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM messages WHERE user_id = ?', (user_id,))
        deleted = cursor.rowcount
        conn.commit()
        logger.info(f"Cleared {deleted} messages for user {user_id}")
        return deleted

try:
    init_database()
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    raise