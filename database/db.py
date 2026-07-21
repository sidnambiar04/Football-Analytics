import os
from pathlib import Path
import psycopg2
from dotenv import load_dotenv

# Always load the .env from the project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def get_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )

        print("✅ Connected to PostgreSQL")
        return conn

    except Exception as e:
        print("❌ Connection Failed")
        print(e)
        return None