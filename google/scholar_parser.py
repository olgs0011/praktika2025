import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import quote
from utils import log_message
import time
from typing import Dict, List, Optional  # Добавьте этот импорт
import json


def search_scholar(query: str, delay: int = 5) -> Optional[str]:
    """Поиск публикаций в Google Scholar по DOI"""
    url = f"https://scholar.google.com/scholar?q={quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        time.sleep(delay)
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        log_message(f"Ошибка запроса к Google Scholar: {str(e)}", "ERROR")
        return None


def process_scholar_results(html: str) -> Dict[str, List[str]]:
    """Анализ результатов Google Scholar и извлечение DOI"""
    soup = BeautifulSoup(html, 'html.parser')
    cited_by_dois = set()
    references_dois = set()

    # 1. Парсим цитирующие работы (Cited by)
    for item in soup.select('.gs_ri'):
        # Извлекаем ссылки на статьи
        for link in item.select('a[href*="doi.org/"]'):
            doi = link['href'].split('doi.org/')[1].split('?')[0].split('#')[0]
            if doi.startswith('10.'):
                cited_by_dois.add(doi)

        # 2. Пытаемся найти DOI в тексте (для ссылок)
        doi_match = re.search(r'doi\.org/(10\.\d+/\S+)', item.text)
        if doi_match:
            references_dois.add(doi_match.group(1))

    return {
        "cited_by": list(cited_by_dois),
        "references": list(references_dois)
    }


# Для тестирования
if __name__ == "__main__":
    test_html = requests.get("https://scholar.google.com/scholar?q=10.5219/923").text
    results = process_scholar_results(test_html)
    print(json.dumps(results, indent=2))