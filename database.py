import sqlite3
from contextlib import contextmanager
from datetime import datetime

DB_NAME = "parking.db"


# ---------- DB CONNECTION ----------
def get_connection():
    return sqlite3.connect(
        DB_NAME,
        check_same_thread=False
    )


# ---------- INIT DB ----------
def init_db():
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS parking_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            plate TEXT NOT NULL,
            province TEXT,

            entry_time TEXT NOT NULL,
            exit_time TEXT,

            duration_minutes INTEGER,
            fee INTEGER,

            entry_image TEXT,
            exit_image TEXT,
            plate_image TEXT
        )
        """)
        conn.commit()


# ---------- INSERT ENTRY ----------
def insert_entry(plate, province, entry_image=None, plate_image=None):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO parking_records
            (plate, province, entry_time, entry_image, plate_image)
            VALUES (?, ?, ?, ?, ?)
        """, (
            plate,
            province,
            datetime.utcnow().isoformat(),
            entry_image,
            plate_image
        ))
        conn.commit()
        return c.lastrowid


# ---------- FIND ACTIVE CAR ----------
def get_active_record(plate):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, entry_time
            FROM parking_records
            WHERE plate = ? AND exit_time IS NULL
            ORDER BY id DESC
            LIMIT 1
        """, (plate,))
        return c.fetchone()


# ---------- EXIT CAR ----------
def close_record(record_id, exit_image=None):
    with get_connection() as conn:
        c = conn.cursor()

        # get entry time
        c.execute("""
            SELECT entry_time
            FROM parking_records
            WHERE id = ?
        """, (record_id,))
        row = c.fetchone()
        if not row:
            return None

        entry_time = datetime.fromisoformat(row[0])
        exit_time = datetime.utcnow()

        minutes = int((exit_time - entry_time).total_seconds() / 60)
        fee = minutes * 1   # rate ต่อ นาที (ปรับได้)

        c.execute("""
            UPDATE parking_records
            SET exit_time = ?,
                duration_minutes = ?,
                fee = ?,
                exit_image = ?
            WHERE id = ?
        """, (
            exit_time.isoformat(),
            minutes,
            fee,
            exit_image,
            record_id
        ))
        conn.commit()

        return {
            "exit_time": exit_time.isoformat(),
            "minutes": minutes,
            "fee": fee
        }


# ---------- GET ALL RECORDS ----------
def get_all_records():
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT
                plate, province,
                entry_time, exit_time,
                duration_minutes, fee
            FROM parking_records
            ORDER BY id DESC
        """)
        return c.fetchall()