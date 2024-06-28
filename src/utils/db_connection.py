import sqlite3
import os
from dotenv import load_dotenv


def db_connection():
    load_dotenv()
    return sqlite3.connect(os.getenv("DB_PATH"))