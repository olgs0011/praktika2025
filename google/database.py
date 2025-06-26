import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os
from typing import List
from utils import log_message

load_dotenv()


def connect_to_scopus_db():
    """Подключение к БД Scopus"""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
        log_message("Подключение к БД установлено", "SUCCESS")
        return conn
    except Exception as e:
        log_message(f"Ошибка подключения: {str(e)}", "ERROR")
        return None

def check_table_structure(conn):
    """Проверка наличия необходимых столбцов в таблице publication"""
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'publication'
                AND column_name IN ('id', 'doi', 'title')
            """)
            columns = [row[0] for row in cursor.fetchall()]
            if len(columns) < 2:  # Минимум id и doi должны быть
                log_message("Таблица publication не содержит нужные столбцы (id, doi)", "ERROR")
                return False
            return True
    except Exception as e:
        log_message(f"Ошибка проверки структуры: {str(e)}", "ERROR")
        return False


def get_publications_batch(conn, start_id: int = 1, limit: int = 10):
    """Получение публикаций (только id и doi)"""
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, doi 
                FROM publication 
                WHERE id >= %s AND doi IS NOT NULL
                ORDER BY id
                LIMIT %s
            """, (start_id, limit))
            return [{'id': row[0], 'doi': row[1]} for row in cursor.fetchall()]
    except Exception as e:
        log_message(f"Ошибка запроса публикаций: {str(e)}", "ERROR")
        return []

def check_table_structure(conn):
    """Упрощенная проверка структуры (только id и doi)"""
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'publication'
                AND column_name IN ('id', 'doi')
            """)
            columns = [row[0] for row in cursor.fetchall()]
            if len(columns) != 2:
                log_message("Таблица publication должна содержать столбцы id и doi", "ERROR")
                return False
            return True
    except Exception as e:
        log_message(f"Ошибка проверки структуры: {str(e)}", "ERROR")
        return False


def get_scopus_ids(conn, urls: List[str]) -> List[str]:
    """Получение scopus_id по URL с DOI из таблицы publication"""
    if not urls:
        return []

    # Извлекаем DOI из URL
    dois = []
    for url in urls:
        if "doi.org/" in url:
            dois.append(url.split("doi.org/")[1].split("?")[0])

    if not dois:
        return []

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id::text 
                FROM publication 
                WHERE doi = ANY(%s)
            """, (dois,))
            return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        log_message(f"Ошибка поиска scopus_id: {str(e)}", "ERROR")
        return []


def save_last_processed_id(last_id: int) -> None:
    """Сохранение последнего обработанного ID"""
    try:
        with open('last_processed.txt', 'w') as f:
            f.write(str(last_id))
    except Exception as e:
        log_message(f"Ошибка сохранения ID: {str(e)}", "ERROR")
