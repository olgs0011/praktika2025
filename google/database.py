import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
import os
from typing import Dict, List, Optional, Union
from utils import log_message

load_dotenv()

BATCH_SIZE = int(os.getenv('BATCH_SIZE', 50))


def connect_to_scopus_db():
    """Устанавливает соединение с базой данных с таймаутом"""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            connect_timeout=10
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        log_message("Успешное подключение к БД", "SUCCESS")
        return conn
    except Exception as e:
        log_message(f"Ошибка подключения к БД: {str(e)}", "ERROR")
        return None


def get_publications_batch(conn, start_id: int = 1, limit: int = BATCH_SIZE) -> List[Dict[str, Union[str, int]]]:
    """Получает публикации, начиная с определённого ID"""
    if not conn or conn.closed:
        log_message("Нет соединения с БД или оно закрыто", "ERROR")
        return []

    try:
        with conn.cursor() as cursor:
            query = """
                SELECT id, doi 
                FROM publication 
                WHERE id >= %s AND doi IS NOT NULL AND doi != ''
                ORDER BY id
                LIMIT %s
            """
            cursor.execute(query, (start_id, limit))

            return [{
                'id': str(row[0]),
                'doi': str(row[1])
            } for row in cursor.fetchall()]
    except Exception as e:
        log_message(f"Ошибка при запросе публикаций: {str(e)}", "ERROR")
        return []


def get_scopus_ids(conn, dois: List[str]) -> List[str]:
    """Находит Scopus ID по списку DOI с пакетной обработкой"""
    if not dois:
        return []

    try:
        with conn.cursor() as cursor:
            query = sql.SQL("""
                SELECT id 
                FROM publication 
                WHERE doi = ANY(%s)
            """)
            cursor.execute(query, (dois,))
            return [str(row[0]) for row in cursor.fetchall()]
    except Exception as e:
        log_message(f"Ошибка поиска Scopus ID: {str(e)}", "ERROR")
        return []


def save_last_processed_id(last_id: int) -> None:
    """Сохраняет последний обработанный ID в файл"""
    try:
        with open('last_processed.txt', 'w') as f:
            f.write(str(last_id))
    except Exception as e:
        log_message(f"Ошибка сохранения последнего ID: {str(e)}", "ERROR")
