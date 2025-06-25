import psycopg2
from typing import Optional, List, Dict, Set
from utils import log_message
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "scopus"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "123456789"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 5))

def connect_to_scopus_db() -> Optional[psycopg2.extensions.connection]:
    """Подключение к базе данных Scopus"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False
        log_message("Успешное подключение к БД Scopus")
        return conn
    except Exception as e:
        log_message(f"Ошибка подключения: {str(e)}", "ERROR")
        return None

def get_publications_batch(conn: psycopg2.extensions.connection, limit: int = BATCH_SIZE) -> List[Dict[str, str]]:
    """Получение пакета публикаций из Scopus"""
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT id, doi FROM publication 
            WHERE doi IS NOT NULL AND doi != ''
            ORDER BY id
            LIMIT %s;
        """, (limit,))
        return [{"id": row[0], "doi": row[1]} for row in cursor.fetchall()]

def get_scopus_ids(conn: psycopg2.extensions.connection, dois: List[str]) -> List[str]:
    """Поиск Scopus ID по списку DOI с нормализацией"""
    if not dois:
        return []

    try:
        with conn.cursor() as cursor:
            # Нормализуем DOI (удаляем URL-часть)
            normalized_dois = [d.split('doi.org/')[-1].split('?')[0] for d in dois if '10.' in d]

            cursor.execute("""
                SELECT id FROM publication 
                WHERE doi = ANY(%s) 
                OR doi LIKE ANY(%s);
            """, (normalized_dois, [f'%{d}%' for d in normalized_dois]))

            return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        log_message(f"Ошибка поиска в Scopus: {str(e)}", "ERROR")
        return []