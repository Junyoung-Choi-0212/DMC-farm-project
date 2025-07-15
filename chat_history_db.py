import sqlite3

def create_chat_history_table(db_path="chat_history.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            question TEXT,
            answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

def insert_chat_history(user_id, question, answer, db_path="chat_history.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO chat_history (user_id, question, answer)
        VALUES (?, ?, ?)
        """,
        (user_id, question, answer)
    )
    conn.commit()
    conn.close()

def load_chat_history(user_id, db_path="chat_history.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT question, answer
        FROM chat_history
        WHERE user_id = ?
        ORDER BY created_at ASC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows  # list of (question, answer)

def get_all_session_ids_with_time(db_path="chat_history.db"):
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_id, MIN(created_at) as started_at
        FROM chat_history
        GROUP BY user_id
        ORDER BY started_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    display_names = [f"{row[1][:16]} | {row[0][:6]}" for row in rows]  # 보기용
    session_ids = [row[0] for row in rows]  # 실제 UUID
    return display_names, session_ids