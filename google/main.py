from typing import Optional, Dict, List, Any
from scholar_parser import get_scholar_data
from database import get_scopus_ids, connect_to_scopus_db, get_publications_batch
from utils import log_message
import time
import random
import json
from datetime import datetime

def main() -> None:
    log_message("Запуск обработки Scopus -> Google Scholar", "INFO")

    # Диагностика подключения
    try:
        conn = connect_to_scopus_db()
        if not conn:
            log_message("Не удалось подключиться к БД", "ERROR")
            return

        with conn.cursor() as cursor:
            # Проверка версии PostgreSQL
            cursor.execute("SELECT version()")
            log_message(f"Версия PostgreSQL: {cursor.fetchone()[0]}", "INFO")

            # Проверка размера таблицы
            cursor.execute("SELECT COUNT(*) FROM publication")
            log_message(f"Всего записей в таблице: {cursor.fetchone()[0]}", "INFO")

            # Проверка первых 5 DOI
            cursor.execute("SELECT doi FROM publication WHERE doi IS NOT NULL LIMIT 5")
            log_message(f"Примеры DOI: {[row[0] for row in cursor.fetchall()]}", "INFO")

    except Exception as e:
        log_message(f"Диагностика не удалась: {str(e)}", "ERROR")
        return
    finally:
        if conn:
            conn.close()

    """Основная функция выполнения"""
    log_message("Запуск обработки Scopus -> Google Scholar", "INFO")

    conn = None
    try:
        conn = connect_to_scopus_db()
        if not conn:
            log_message("Не удалось установить соединение с БД", "ERROR")
            return

        # Проверка подключения
        with conn.cursor() as cursor:
            cursor.execute("SELECT current_database(), current_user")
            db_info = cursor.fetchone()
            log_message(f"Подключено к БД: {db_info[0]} как пользователь: {db_info[1]}", "INFO")

        publications = get_publications_batch(conn)
        log_message(f"Получено публикаций: {len(publications)}", "INFO")

        results = []
        for pub in publications:
            try:
                log_message(f"Обрабатываю публикацию ID: {pub['id']}, DOI: {pub['doi']}", "DEBUG")

                # Получаем данные из Google Scholar
                scholar_data = get_scholar_data(pub['doi'])
                if not scholar_data:
                    log_message(f"Не удалось получить данные Scholar для DOI: {pub['doi']}", "WARNING")
                    continue

                # Находим Scopus ID для цитирований и ссылок
                cited_by_ids = get_scopus_ids(conn, scholar_data['cited_by'])
                cites_ids = get_scopus_ids(conn, scholar_data['references'])

                results.append({
                    'scopus_id': pub['id'],
                    'doi': pub['doi'],
                    'cited_by': cited_by_ids,
                    'cites': cites_ids,
                    'scholar_data': scholar_data
                })

                # Искусственная задержка
                time.sleep(random.uniform(1, 3))

            except Exception as e:
                log_message(f"Ошибка обработки публикации {pub.get('id', 'N/A')}: {str(e)}", "ERROR")
                continue

        # Сохранение результатов
        if results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'results_{timestamp}.json'

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            log_message(f"Результаты сохранены в {filename}", "SUCCESS")

    except Exception as e:
        log_message(f"Фатальная ошибка: {str(e)}", "ERROR")
    finally:
        if conn:
            conn.close()
            log_message("Соединение закрыто", "INFO")


if __name__ == "__main__":
    main()
