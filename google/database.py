import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT  # Добавьте этот импорт
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Union
from utils import log_message

load_dotenv()

# Константы
BATCH_SIZE = 5
DB_CONNECT_TIMEOUT = 10


def connect_to_scopus_db():
    """Подключение к базе данных с обработкой ошибок"""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            connect_timeout=DB_CONNECT_TIMEOUT
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        log_message("Успешное подключение к БД", "SUCCESS")
        return conn
    except Exception as e:
        log_message(f"Ошибка подключения к БД: {str(e)}", "ERROR")
        return None


def get_publications_batch(conn, start_id: int = 1, limit: int = BATCH_SIZE) -> List[Dict[str, Union[str, int]]]:
    """Получение пакета публикаций"""
    if not conn or conn.closed:
        log_message("Соединение с БД не установлено", "ERROR")
        return []

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id::text, doi::text 
                FROM publication 
                WHERE id >= %s AND doi IS NOT NULL
                ORDER BY id
                LIMIT %s
            """, (start_id, limit))

            return [{
                'id': str(row[0]),
                'doi': str(row[1])
            } for row in cursor.fetchall()]
    except Exception as e:
        log_message(f"Ошибка при запросе публикаций: {str(e)}", "ERROR")
        return []


def get_scopus_ids(conn, dois: List[str]) -> List[str]:
    """Получение scopus_id по DOI"""
    if not dois:
        return []

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id::text 
                FROM publication 
                WHERE doi = ANY(%s)
            """, (dois,))
            return [str(row[0]) for row in cursor.fetchall()]
    except Exception as e:
        log_message(f"Ошибка поиска scopus_id: {str(e)}", "ERROR")
        return []


def save_last_processed_id(last_id: int) -> None:
    """Сохранение последнего обработанного ID"""
    try:
        with open('last_processed.txt', 'w') as f:
            f.write(str(last_id))
    except Exception as e:
        log_message(f"Ошибка сохранения last_processed_id: {str(e)}", "ERROR")
