from typing import Dict, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import quote, unquote
import time
import random
from utils import log_message
import json

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Referer': 'https://scholar.google.com/',
}

PROXY_LIST = [
    None,  # Прямое подключение тоже пробуем
    # Добавьте свои прокси здесь при необходимости
]

REQUEST_DELAY = (1, 3)  # Уменьшенные задержки для тестирования
MAX_RETRIES = 3


def extract_doi_from_text(text: str) -> Optional[str]:
    """Извлекает DOI из текста"""
    if not text:
        return None

    doi_patterns = [
        r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b',
        r'doi:\s*(10\.\d{4,9}/[-._;()/:A-Z0-9]+)',
        r'DOI\s*=\s*(10\.\d{4,9}/[-._;()/:A-Z0-9]+)'
    ]

    for pattern in doi_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            doi = match.group(1)
            return doi.lower().strip()
    return None


def extract_doi_from_url(url: str) -> Optional[str]:
    """Извлекает DOI из URL"""
    if not url:
        return None

    url = unquote(url).lower()

    patterns = [
        r'doi\.org/(10\.\d{4,9}/[-._;()/:a-z0-9]+)',
        r'doi\.org/(10\.\d{4,9})',
        r'[?&;]doi=(10\.\d{4,9}/[-._;()/:a-z0-9]+)',
        r'citation_doi=(10\.\d{4,9}/[-._;()/:a-z0-9]+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            doi = match.group(1)
            return doi.strip()
    return None


def parse_scholar_page(html: str) -> Tuple[List[str], List[str]]:
    """Парсинг HTML страницы Google Scholar"""
    soup = BeautifulSoup(html, 'html.parser')
    cited_by = set()
    references = set()

    for item in soup.select('.gs_ri'):
        for link in item.select('a[href]'):
            if doi := extract_doi_from_url(link['href']):
                cited_by.add(doi)

        if doi := extract_doi_from_text(item.get_text()):
            references.add(doi)

    return list(cited_by), list(references)


def make_scholar_request(url: str) -> Optional[requests.Response]:
    """Выполняет запрос к Google Scholar с ротацией прокси"""
    proxies = None
    if PROXY_LIST:
        proxies = random.choice(PROXY_LIST)

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            proxies=proxies,
            timeout=30
        )
        response.raise_for_status()

        if 'sorry' in response.text.lower() or 'captcha' in response.text.lower():
            raise ValueError("Обнаружена CAPTCHA")

        return response

    except Exception as e:
        log_message(f"Ошибка запроса: {str(e)}", "ERROR")
        return None


def get_scholar_data(doi: str) -> Dict[str, List[str]]:
    """Основная функция получения данных из Google Scholar"""
    if not doi or not isinstance(doi, str) or not doi.strip():
        log_message(f"Некорректный DOI: {doi}", "WARNING")
        return {'cited_by': [], 'references': []}

    url = f"https://scholar.google.com/scholar?hl=en&q={quote(doi)}"
    result = {'cited_by': [], 'references': []}

    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(random.uniform(*REQUEST_DELAY))
            log_message(f"Попытка {attempt + 1}/{MAX_RETRIES} для DOI: {doi}", "DEBUG")

            response = make_scholar_request(url)
            if not response:
                continue

            cited_by, references = parse_scholar_page(response.text)
            result['cited_by'] = cited_by
            result['references'] = references

            log_message(f"Найдено {len(cited_by)} цитирований и {len(references)} ссылок для DOI: {doi}", "SUCCESS")
            return result

        except Exception as e:
            log_message(f"Ошибка при обработке DOI {doi}: {str(e)}", "ERROR")
            if attempt == MAX_RETRIES - 1:
                log_message(f"Не удалось обработать DOI после {MAX_RETRIES} попыток: {doi}", "WARNING")

    return result
