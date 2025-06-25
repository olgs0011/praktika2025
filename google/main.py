from database import connect_to_scopus_db, get_publications_batch, get_scopus_ids
from scholar_parser import search_scholar, process_scholar_results
from utils import log_message
import time
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def save_results(results: list, filename: str = None):
    """Сохранение результатов в JSON файл в требуемом формате"""
    if not results:
        log_message("Нет результатов для сохранения", "WARNING")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = filename or f"scopus_scholar_relations_{timestamp}.json"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log_message(f"Результаты сохранены в {filename}")
    except Exception as e:
        log_message(f"Ошибка сохранения: {str(e)}", "ERROR")


def main():
    log_message("Запуск обработки: Scopus + Google Scholar")
    results = []

    conn = connect_to_scopus_db()
    if not conn:
        return

    try:
        publications = get_publications_batch(conn)
        if not publications:
            log_message("Нет публикаций для обработки", "WARNING")
            return

        for pub in publications:
            log_message(f"\nОбработка публикации Scopus ID: {pub['id']}, DOI: {pub['doi']}")

            # Получаем данные из Google Scholar
            scholar_data = search_scholar(pub['doi'])

            if not scholar_data:
                log_message("Не найдено результатов в Google Scholar", "WARNING")
                continue

            # Обрабатываем результаты для получения DOI цитирующих и цитируемых работ
            relations = process_scholar_results(scholar_data)

            # Получаем Scopus ID для связанных публикаций
            cited_by_ids = get_scopus_ids(conn, relations["cited_by"])
            cites_ids = get_scopus_ids(conn, relations["references"])

            result = {
                "query_id": pub["id"],
                "cites": cites_ids,
                "cited_by": cited_by_ids
            }

            results.append(result)
            log_message(f"Результат: {json.dumps(result, indent=2)}")

            time.sleep(int(os.getenv("REQUEST_DELAY", 5)))

    finally:
        conn.close()
        save_results(results)
        log_message(f"Обработано публикаций: {len(results)}")
        log_message("Работа завершена")


if __name__ == "__main__":
    main()