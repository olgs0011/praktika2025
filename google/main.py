from typing import List, Dict, Optional
from selenium_scholar import get_scholar_data
from database import (
    connect_to_scopus_db,
    get_publications_batch,
    get_scopus_ids,
    save_last_processed_id
)
from utils import log_message
import json
import time
import random
from dotenv import load_dotenv

load_dotenv()


def save_results_to_json(results: List[Dict], filename: str = "scholar_results.json"):
    """Сохранение результатов в JSON файл"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log_message(f"Результаты сохранены в {filename}", "SUCCESS")
    except Exception as e:
        log_message(f"Ошибка при сохранении JSON: {str(e)}", "ERROR")


def process_publication(conn, pub: Dict) -> Optional[Dict]:
    """Обработка одной публикации"""
    doi = pub.get('doi')
    if not doi:
        log_message(f"Пропуск публикации {pub['id']} - нет DOI", "WARNING")
        return None

    log_message(f"Начата обработка ID: {pub['id']}, DOI: {doi}", "INFO")

    scholar_data = get_scholar_data(doi)
    if not scholar_data:
        log_message(f"Не удалось получить данные для ID: {pub['id']}", "WARNING")
        return None

    # Получаем Scopus ID для цитирований
    cited_by_ids = get_scopus_ids(conn, scholar_data.get('cited_by', []))
    cites_ids = get_scopus_ids(conn, scholar_data.get('references', []))

    return {
        'query_id': pub['id'],
        'cited_by': cited_by_ids,
        'cites': cites_ids
    }


def main():
    """Основной рабочий процесс"""
    log_message("Запуск парсера Scopus -> Google Scholar", "INFO")
    conn = connect_to_scopus_db()
    if not conn:
        return

    results = []
    try:
        last_id = 1  # Можно заменить на загрузку из last_processed.txt
        while True:
            publications = get_publications_batch(conn, start_id=last_id)
            if not publications:
                log_message("Нет новых публикаций для обработки", "INFO")
                break

            for pub in publications:
                result = process_publication(conn, pub)
                if result:
                    results.append(result)
                    save_last_processed_id(int(pub['id']))
                    last_id = int(pub['id'])

                    # Сохраняем промежуточные результаты
                    save_results_to_json(results)

                # Случайная задержка между запросами
                delay = random.uniform(30, 120)  # Увеличьте при блокировках
                log_message(f"Пауза {delay:.1f} сек перед следующим запросом", "DEBUG")
                time.sleep(delay)

    except KeyboardInterrupt:
        log_message("Остановлено пользователем", "INFO")
    except Exception as e:
        log_message(f"Критическая ошибка: {str(e)}", "ERROR")
    finally:
        conn.close()
        save_results_to_json(results)
        log_message("Обработка завершена. Итоговые результаты сохранены.", "INFO")


if __name__ == "__main__":
    main()
