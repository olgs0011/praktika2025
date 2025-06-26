from typing import Optional, Dict, List, Any
from scholar_parser import get_scholar_data
from database import get_scopus_ids, connect_to_scopus_db, get_publications_batch, save_last_processed_id
from utils import log_message
import time
import random
import json
from datetime import datetime
import os


def load_last_processed_id() -> int:
    """Загружает последний обработанный ID из файла"""
    try:
        if os.path.exists('last_processed.txt'):
            with open('last_processed.txt', 'r') as f:
                content = f.read().strip()
                return int(content) if content else 1
    except Exception as e:
        log_message(f"Ошибка загрузки last_processed_id: {str(e)}", "ERROR")
    return 1


def process_publications(conn, start_id: int) -> List[Dict[str, Any]]:
    """Обрабатывает публикации и возвращает результаты"""
    results = []
    last_id = start_id

    publications = get_publications_batch(conn, start_id=last_id)
    if not publications:
        log_message("Не найдено публикаций для обработки", "INFO")
        return results

    for pub in publications:
        try:
            log_message(f"Обрабатываю публикацию ID: {pub['id']}, DOI: {pub['doi']}", "DEBUG")
            last_id = int(pub['id'])

            scholar_data = get_scholar_data(pub['doi'])
            if scholar_data is None:
                log_message(f"Публикация {pub['id']} не найдена в Google Scholar, пропускаем", "INFO")
                save_last_processed_id(last_id)
                time.sleep(random.uniform(1, 3))
                continue

            cited_by_ids = get_scopus_ids(conn, scholar_data['cited_by'])
            cites_ids = get_scopus_ids(conn, scholar_data['references'])

            results.append({
                'scopus_id': pub['id'],
                'doi': pub['doi'],
                'cited_by': cited_by_ids,
                'cites': cites_ids,
                'scholar_data': scholar_data
            })

            save_last_processed_id(last_id)
            time.sleep(random.uniform(1, 3))

        except Exception as e:
            log_message(f"Ошибка обработки публикации {pub.get('id', 'N/A')}: {str(e)}", "ERROR")
            continue

    return results


def main() -> None:
    """Основная функция выполнения"""
    log_message("Запуск обработки Scopus -> Google Scholar", "INFO")

    conn = None
    try:
        conn = connect_to_scopus_db()
        if not conn:
            log_message("Не удалось установить соединение с БД", "ERROR")
            return

        last_id = load_last_processed_id()
        log_message(f"Начинаем обработку с ID: {last_id}", "INFO")

        results = process_publications(conn, last_id)
        log_message(f"Обработано публикаций: {len(results)}", "INFO")

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
