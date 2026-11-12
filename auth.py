"""Small local authentication layer for the Streamlit application."""

import hashlib
import hmac
import os
import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "iris_app.db"
PBKDF2_ITERATIONS = 310_000


def _connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return connection


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def register_user(full_name: str, email: str, password: str) -> tuple[bool, str]:
    """Create a local account and return a user-facing status message."""
    full_name, email = full_name.strip(), email.strip().lower()
    if not full_name or not email or not password:
        return False, "Please complete every field."
    if "@" not in email:
        return False, "Enter a valid email address."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."

    try:
        with _connection() as connection:
            connection.execute(
                "INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
                (full_name, email, _hash_password(password)),
            )
    except sqlite3.IntegrityError:
        return False, "An account already exists for that email address."
    return True, "Account created. Please sign in."


def authenticate_user(email: str, password: str) -> dict | None:
    """Return the account details when credentials are valid."""
    with _connection() as connection:
        row = connection.execute(
            "SELECT full_name, email, password_hash FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()
    if row is None:
        return None

    salt_hex, expected_hash = row[2].split("$", 1)
    received_hash = _hash_password(password, bytes.fromhex(salt_hex)).split("$", 1)[1]
    if not hmac.compare_digest(received_hash, expected_hash):
        return None
    return {"name": row[0], "email": row[1]}
