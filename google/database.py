import psycopg2
from psycopg2 import extensions
from dotenv import load_dotenv
import os
from typing import Dict, List, Optional
from utils import log_message

load_dotenv()

BATCH_SIZE = int(os.getenv('BATCH_SIZE', 5))


def connect_to_scopus_db():
    """Устанавливает соединение с базой данных"""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
        log_message("Успешное подключение к БД", "SUCCESS")
        return conn
    except Exception as e:
        log_message(f"Ошибка подключения к БД: {str(e)}", "ERROR")
        return None


def get_publications_batch(conn, limit=BATCH_SIZE):
    """Полностью переработанная функция с максимальной защитой"""
    if not conn or conn.closed:
        log_message("Нет соединения с БД или оно закрыто", "ERROR")
        return []

    try:
        with conn.cursor() as cursor:
            # Упрощенный запрос без условий для теста
            cursor.execute("SELECT id::text, doi::text FROM publication LIMIT %s", (limit,))

            results = []
            for record in cursor:
                try:
                    # Самый безопасный способ обработки
                    record_dict = dict(zip([desc[0] for desc in cursor.description], record))
                    if 'id' in record_dict and 'doi' in record_dict:
                        results.append({
                            'id': str(record_dict['id']),
                            'doi': str(record_dict['doi'])
                        })
                except Exception as e:
                    log_message(f"Ошибка обработки записи: {str(e)}", "DEBUG")
                    continue

            log_message(f"Получено {len(results)} записей", "INFO")
            return results

    except Exception as e:
        log_message(f"Критическая ошибка при запросе: {str(e)}", "ERROR")
        return []

def get_scopus_ids(conn: psycopg2.extensions.connection, dois: List[str]) -> List[str]:
    """Находит Scopus ID по списку DOI"""
    if not dois:
        return []

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id 
                FROM publication 
                WHERE doi = ANY(%s)
            """, (dois,))
            return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        log_message(f"Ошибка поиска Scopus ID: {str(e)}", "ERROR")
        return []
