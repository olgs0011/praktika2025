from typing import Optional, List, Dict, Set, Tuple
import psycopg2
import json
import requests
from urllib.parse import quote, unquote
import time
from datetime import datetime

# Конфигурация
DB_CONFIG = {
    "dbname": "scopus",
    "user": "postgres",
    "password": "123456789",
    "host": "localhost",
    "port": "5432"
}

OPENALEX_EMAIL = "SeVaStUnovA20056@yandex.ru"
REQUEST_DELAY = 2
MAX_PAGES = 200
BATCH_SIZE = 5
DEBUG_MODE = True  # Режим отладки с дополнительным логированием


def log_message(message: str, level: str = "INFO"):
    """Улучшенное логирование с уровнями важности"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}")


def normalize_doi(doi: str) -> str:
    """Приводит DOI к стандартному формату без URL-кодирования"""
    if not doi:
        return ""

    # Удаляем кодирование URL и лишние пробелы
    doi = unquote(doi).strip().lower()

    # Удаляем префикс 'https://doi.org/' если есть
    if doi.startswith('https://doi.org/'):
        doi = doi[16:]
    elif doi.startswith('doi.org/'):
        doi = doi[7:]

    # Удаляем возможные параметры после DOI
    doi = doi.split('?')[0].split('#')[0].split(' ')[0]

    # Убедимся, что DOI начинается с '10.'
    if not doi.startswith('10.'):
        log_message(f"Нестандартный DOI формат: {doi}", "WARNING")

    return doi


def connect_to_scopus_db() -> Optional[psycopg2.extensions.connection]:
    """Подключение к базе данных с обработкой ошибок"""
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


def safe_api_request(url: str, max_retries: int = 3) -> Optional[requests.Response]:
    """Безопасный запрос к API с повторными попытками"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30, headers={"User-Agent": f"mailto:{OPENALEX_EMAIL}"})

            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 30))
                log_message(f"Достигнут лимит запросов. Пауза {retry_after} сек...", "WARNING")
                time.sleep(retry_after)
                continue

            if response.status_code == 200:
                return response

            log_message(f"HTTP {response.status_code} для URL: {url}", "WARNING")
            time.sleep(5)

        except Exception as e:
            log_message(f"Ошибка запроса (попытка {attempt + 1}): {str(e)}", "ERROR")
            time.sleep(5)

    return None


def get_all_related_works(api_url: str) -> Tuple[List[Dict], int]:
    """Получение всех связанных работ с пагинацией"""
    works = []
    next_cursor = None
    total_pages = 0

    for _ in range(MAX_PAGES):
        current_url = f"{api_url}&cursor={next_cursor}" if next_cursor else api_url
        response = safe_api_request(current_url)

        if not response:
            break

        data = response.json()
        works.extend(data.get("results", []))
        total_pages += 1

        if not data.get("meta", {}).get("next_cursor"):
            break

        next_cursor = data["meta"]["next_cursor"]
        time.sleep(REQUEST_DELAY)

    return works, total_pages


def extract_doi_from_work(work: Dict) -> Optional[str]:
    """Извлечение и нормализация DOI из работы OpenAlex"""
    if not work.get('doi'):
        return None

    doi = normalize_doi(work['doi'])
    if not doi.startswith('10.'):
        log_message(f"Найден нестандартный DOI: {doi}", "DEBUG")
        return None

    return doi


def process_references(work_ids: List[str]) -> Set[str]:
    """Обработка списка ссылок с прогресс-логгированием"""
    dois = set()
    total = len(work_ids)
    processed = 0

    for work_id in work_ids:
        url = f"https://api.openalex.org/{work_id}?mailto={OPENALEX_EMAIL}"
        response = safe_api_request(url)

        if response:
            data = response.json()
            if doi := extract_doi_from_work(data):
                dois.add(doi)

        processed += 1
        if processed % 10 == 0 or processed == total:
            log_message(f"Обработано ссылок: {processed}/{total} ({len(dois)} уникальных DOI)")

        time.sleep(REQUEST_DELAY)

    return dois


def get_openalex_relations(doi: str) -> Dict:
    """Получение информации о связях из OpenAlex"""
    try:
        norm_doi = normalize_doi(doi)
        if not norm_doi.startswith('10.'):
            return {"error": f"Невалидный DOI формат: {doi}"}

        work_url = f"https://api.openalex.org/works/https://doi.org/{quote(norm_doi)}?mailto={OPENALEX_EMAIL}"
        log_message(f"Запрос к OpenAlex для DOI: {norm_doi}", "DEBUG")

        response = safe_api_request(work_url)
        if not response:
            return {"error": "Статья не найдена в OpenAlex"}

        work_data = response.json()
        if DEBUG_MODE:
            with open(f"debug_{norm_doi.replace('/', '_')}.json", 'w') as f:
                json.dump(work_data, f, indent=2)

        log_message(f"Обрабатывается: {work_data.get('display_name')} (ID: {work_data.get('id')})")

        # Обработка цитирующих работ
        cited_by_dois = set()
        if cited_by_api := work_data.get("cited_by_api_url"):
            cited_by_api += f"&mailto={OPENALEX_EMAIL}"
            cited_by_works, pages = get_all_related_works(cited_by_api)
            cited_by_dois.update(filter(None, map(extract_doi_from_work, cited_by_works)))
            log_message(f"Найдено {len(cited_by_dois)} цитирующих работ (страниц: {pages})")

        # Обработка ссылок
        references_dois = process_references(work_data.get("referenced_works", []))
        log_message(f"Найдено {len(references_dois)} цитируемых работ")

        return {
            "cited_by": list(cited_by_dois),
            "references": list(references_dois),
            "openalex_data": {
                "id": work_data.get("id"),
                "title": work_data.get("display_name"),
                "publication_year": work_data.get("publication_year"),
                "normalized_doi": norm_doi
            }
        }

    except Exception as e:
        log_message(f"Ошибка обработки DOI {doi}: {str(e)}", "ERROR")
        return {"error": str(e)}


def get_scopus_ids(conn: psycopg2.extensions.connection, dois: List[str]) -> List[str]:
    """Поиск Scopus ID по списку DOI"""
    if not dois:
        return []

    normalized_dois = [normalize_doi(d) for d in dois if normalize_doi(d).startswith('10.')]
    if not normalized_dois:
        return []

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id FROM publication 
                WHERE doi = ANY(%s);
            """, (normalized_dois,))
            return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        log_message(f"Ошибка поиска в Scopus: {str(e)}", "ERROR")
        return []


def process_publication(conn: psycopg2.extensions.connection, publication: Dict) -> Optional[Dict]:
    """Обработка одной публикации"""
    pub_id, doi = publication["id"], publication["doi"]
    log_message(f"\nНачата обработка публикации Scopus ID: {pub_id} (DOI: {doi})")

    relations = get_openalex_relations(doi)
    if "error" in relations:
        log_message(f"Ошибка для публикации {pub_id}: {relations['error']}", "WARNING")
        return None

    cited_by_ids = get_scopus_ids(conn, relations["cited_by"])
    references_ids = get_scopus_ids(conn, relations["references"])

    result = {
        "query_id": pub_id,
        "query_doi": doi,
        "cites": references_ids,
        "cited_by": cited_by_ids,
        }

    if DEBUG_MODE:
        with open(f"result_{pub_id}.json", 'w') as f:
            json.dump(result, f, indent=2)

    return result


def save_results(results: List[Dict], filename: str = None):
    """Сохранение результатов в JSON файл"""
    if not results:
        log_message("Нет результатов для сохранения", "WARNING")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = filename or f"scopus_relations_{timestamp}.json"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log_message(f"Результаты сохранены в {filename}")
    except Exception as e:
        log_message(f"Ошибка сохранения результатов: {str(e)}", "ERROR")


def analyze_results(results: List[Dict]):
    """Анализ и статистика по результатам"""
    if not results:
        return

    total = len(results)
    success = sum(1 for r in results if r is not None)
    found_in_oa = sum(1 for r in results if r and "openalex_id" in r)

    avg_cited_by = sum(len(r["cited_by"]) for r in results if r) / success if success else 0
    avg_references = sum(len(r["cites"]) for r in results if r) / success if success else 0

    log_message("\n=== Анализ результатов ===")
    log_message(f"Обработано публикаций: {total}")
    log_message(f"Успешно обработано: {success} ({success / total * 100:.1f}%)")
    log_message(f"Найдено в OpenAlex: {found_in_oa} ({found_in_oa / total * 100:.1f}%)")
    log_message(f"Среднее цитирований на публикацию: {avg_cited_by:.1f}")
    log_message(f"Среднее ссылок на публикацию: {avg_references:.1f}")


def main():
    """Основная функция выполнения"""
    log_message("Запуск обработки Scopus -> OpenAlex")
    conn = connect_to_scopus_db()
    if not conn:
        return

    results = []

    try:
        publications = get_publications_batch(conn)
        if not publications:
            log_message("Нет публикаций с DOI для обработки", "WARNING")
            return

        log_message(f"Начата обработка {len(publications)} публикаций")

        for pub in publications:
            result = process_publication(conn, pub)
            results.append(result)
            time.sleep(REQUEST_DELAY)

            # Периодическое сохранение промежуточных результатов
            if len(results) % 10 == 0:
                save_results([r for r in results if r], f"partial_{len(results)}.json")

        # Фильтрация None результатов перед сохранением
        valid_results = [r for r in results if r]
        save_results(valid_results)
        analyze_results(valid_results)

    except KeyboardInterrupt:
        log_message("Обработка прервана пользователем", "WARNING")
        if results:
            save_results([r for r in results if r], "interrupted_results.json")
    except Exception as e:
        log_message(f"Критическая ошибка: {str(e)}", "ERROR")
    finally:
        conn.close()
        log_message("Работа завершена")


if __name__ == "__main__":
    main()