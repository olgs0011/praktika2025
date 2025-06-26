from typing import Dict, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import quote, unquote
import time
import random
from utils import log_message
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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

PROXY_LIST = []

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (Linux; Android 10; SM-A505FN) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.210 Mobile Safari/537.36",
]

REQUEST_DELAY = (1, 3)
MAX_RETRIES = 3
SELENIUM_TIMEOUT = 30


def make_scholar_request(url: str) -> Optional[requests.Response]:
    """Выполняет запрос к Google Scholar с рандомизацией"""
    try:
        headers = HEADERS.copy()
        headers['User-Agent'] = random.choice(USER_AGENTS)

        # Добавляем случайные задержки между запросами
        time.sleep(random.uniform(*REQUEST_DELAY))

        response = requests.get(
            url,
            headers=headers,
            proxies=random.choice(PROXY_LIST) if PROXY_LIST else None,
            timeout=30
        )
        response.raise_for_status()

        # Проверка на блокировку
        if "Our systems have detected unusual traffic" in response.text:
            log_message("Обнаружена блокировка Google Scholar", "WARNING")
            return None

        return response
    except Exception as e:
        log_message(f"Ошибка запроса: {str(e)}", "ERROR")
        return None


def extract_doi_from_text(text: str) -> Optional[str]:
    """Извлекает DOI из текста с улучшенными регулярками"""
    if not text:
        return None

    doi_patterns = [
        r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b',
        r'doi:\s*(10\.\d{4,9}/[-._;()/:A-Z0-9]+)',
        r'DOI\s*=\s*(10\.\d{4,9}/[-._;()/:A-Z0-9]+)',
        r'doi\.org/(10\.\d{4,9}/[-._;()/:A-Z0-9]+)'
    ]

    for pattern in doi_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            doi = match.group(1)
            # Нормализация DOI
            doi = doi.lower().strip().replace(' ', '')
            if not doi.startswith('http'):
                doi = f'https://doi.org/{doi}'
            return doi
    return None


def extract_doi_from_url(url: str) -> Optional[str]:
    """Извлекает DOI из URL с улучшенной обработкой"""
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
            return f'https://doi.org/{doi.strip()}'
    return None


def parse_scholar_page(html: str) -> Tuple[List[str], List[str]]:
    """Улучшенный парсинг страницы Google Scholar"""
    if "did not match any articles" in html:
        return [], []

    if "Our systems have detected unusual traffic" in html:
        log_message("Обнаружена капча или блокировка", "WARNING")
        return [], []

    soup = BeautifulSoup(html, 'html.parser')
    cited_by = set()
    references = set()

    # Основной блок с результатами
    results = soup.select('.gs_r.gs_or.gs_scl, .gs_ri')

    if not results:
        log_message("Не найдено результатов на странице", "DEBUG")
        return [], []

    for item in results:
        # Извлекаем DOI из текста результата
        text = item.get_text()
        if doi := extract_doi_from_text(text):
            references.add(doi)

        # Извлекаем DOI из ссылок
        for link in item.select('a[href]'):
            href = link['href']
            if doi := extract_doi_from_url(href):
                if "cites" in href or "citedby" in href:
                    cited_by.add(doi)
                else:
                    references.add(doi)

        # Проверяем блок "Цитируется" (Cited by)
        cited_block = item.select_one('.gs_fl a[href*="cites"]')
        if cited_block and (doi := extract_doi_from_url(cited_block['href'])):
            cited_by.add(doi)

    return list(cited_by), list(references)


def get_scholar_data_with_selenium(doi: str) -> Optional[Dict[str, List[str]]]:
    """Альтернативный метод с использованием Selenium"""
    options = Options()
    options.add_argument("--headless=new")  # Новый режим headless
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
    options.add_argument("--disable-blink-features=AutomationControlled")

    service = webdriver.ChromeService()
    driver = None
    try:
        driver = webdriver.Chrome(options=options, service=service)
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": random.choice(USER_AGENTS)
        })

        url = f"https://scholar.google.com/scholar?hl=en&q={quote(doi)}"
        driver.get(url)

        # Проверка на капчу сразу
        if "Our systems have detected unusual traffic" in driver.page_source:
            log_message("Обнаружена капча при использовании Selenium", "WARNING")
            return None

        # Ждем либо результаты, либо сообщение об отсутствии результатов
        try:
            WebDriverWait(driver, 10).until(
                lambda d: d.find_elements(By.CSS_SELECTOR, ".gs_ri, .gs_r, .gs_or") or
                          "did not match any articles" in d.page_source
            )
        except Exception as e:
            log_message(f"Таймаут ожидания результатов: {str(e)}", "WARNING")
            return None

        # Проверка на отсутствие результатов
        if "did not match any articles" in driver.page_source:
            return None

        cited_by, references = parse_scholar_page(driver.page_source)
        return {'cited_by': cited_by, 'references': references}

    except Exception as e:
        log_message(f"Selenium ошибка: {str(e)}", "ERROR")
        return None
    finally:
        if driver:
            driver.quit()

def get_scholar_data(doi: str) -> Optional[Dict[str, List[str]]]:
    """Основная функция получения данных из Google Scholar"""
    if not doi or not isinstance(doi, str) or not doi.strip():
        log_message(f"Некорректный DOI: {doi}", "WARNING")
        return None  # Возвращаем None вместо пустого словаря

    # Нормализация DOI
    if not doi.startswith('http'):
        doi = f'https://doi.org/{doi}'

    # Проверка доступности статьи в Scholar
    check_url = f"https://scholar.google.com/scholar?hl=en&q={quote(doi)}"
    check_response = make_scholar_request(check_url)
    if not check_response or "did not match any articles" in check_response.text:
        log_message(f"Статья с DOI {doi} не найдена в Google Scholar", "INFO")
        return None

    url = f"https://scholar.google.com/scholar?hl=en&q={quote(doi)}"
    result = {'cited_by': [], 'references': []}

    for attempt in range(MAX_RETRIES):
        try:
            log_message(f"Попытка {attempt + 1}/{MAX_RETRIES} для DOI: {doi}", "DEBUG")

            # Сначала пробуем обычный запрос
            response = make_scholar_request(url)

            if response:
                # Сохраняем HTML для отладки
                with open(f"debug_{doi.replace('/', '_')}.html", "w", encoding="utf-8") as f:
                    f.write(response.text)

                cited_by, references = parse_scholar_page(response.text)
                result['cited_by'] = cited_by
                result['references'] = references

                if cited_by or references:
                    log_message(f"Найдено {len(cited_by)} цитирований и {len(references)} ссылок для DOI: {doi}",
                                "SUCCESS")
                    return result
                else:
                    log_message(f"Не найдено результатов, пробуем Selenium", "DEBUG")

            # Если обычный запрос не дал результатов, пробуем Selenium
            selenium_data = get_scholar_data_with_selenium(doi)
            if selenium_data['cited_by'] or selenium_data['references']:
                log_message(
                    f"Selenium нашел {len(selenium_data['cited_by'])} цитирований и {len(selenium_data['references'])} ссылок",
                    "SUCCESS")
                return selenium_data

        except Exception as e:
            log_message(f"Ошибка при обработке DOI {doi}: {str(e)}", "ERROR")
            if attempt == MAX_RETRIES - 1:
                log_message(f"Не удалось обработать DOI после {MAX_RETRIES} попыток: {doi}", "WARNING")

    return result
