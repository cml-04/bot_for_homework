import sqlite3
import hashlib
import os
import uuid


class UserDatabase:
    def __init__(self, db_path="user_database.db"):
        self.db_path = db_path
        self.create_tables()

    def create_tables(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            api_key TEXT,
            reset_token TEXT,
            reset_token_expiry TEXT
        )
        ''')

        conn.commit()
        conn.close()

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def register_user(self, username, password, email, api_key=""):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            password_hash = self.hash_password(password)
            cursor.execute(
                "INSERT INTO users (username, password_hash, email, api_key) VALUES (?, ?, ?, ?)",
                (username, password_hash, email, api_key)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def authenticate(self, username, password):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        password_hash = self.hash_password(password)
        cursor.execute(
            "SELECT id, username, api_key, email FROM users WHERE username = ? AND password_hash = ?",
            (username, password_hash)
        )
        user = cursor.fetchone()
        conn.close()

        if user:
            return {"id": user[0], "username": user[1], "api_key": user[2], "email": user[3]}
        return None

    def update_api_key(self, username, api_key):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE users SET api_key = ? WHERE username = ?",
            (api_key, username)
        )
        conn.commit()
        conn.close()

    def generate_reset_token(self, email):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if email exists
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        if not cursor.fetchone():
            conn.close()
            return None

        # Generate token
        token = str(uuid.uuid4())
        import datetime
        expiry = datetime.datetime.now() + datetime.timedelta(hours=1)
        expiry_str = expiry.isoformat()

        cursor.execute(
            "UPDATE users SET reset_token = ?, reset_token_expiry = ? WHERE email = ?",
            (token, expiry_str, email)
        )
        conn.commit()
        conn.close()

        return token

    def reset_password(self, token, new_password):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        import datetime
        now = datetime.datetime.now().isoformat()

        cursor.execute(
            "SELECT id FROM users WHERE reset_token = ? AND reset_token_expiry > ?",
            (token, now)
        )
        user = cursor.fetchone()

        if not user:
            conn.close()
            return False

        password_hash = self.hash_password(new_password)
        cursor.execute(
            "UPDATE users SET password_hash = ?, reset_token = NULL, reset_token_expiry = NULL WHERE id = ?",
            (password_hash, user[0])
        )
        conn.commit()
        conn.close()

        return True