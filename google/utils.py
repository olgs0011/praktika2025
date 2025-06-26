from datetime import datetime
from typing import Any, Optional
import sys
import re
import logging
from pathlib import Path


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def setup_logging():
    """Настройка логирования в файл и консоль"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'scopus_scholar.log'),
            logging.StreamHandler()
        ]
    )


def log_message(message: str, level: str = "INFO") -> None:
    """Улучшенное логирование с цветами и записью в файл"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    level = level.upper()

    color = {
        "INFO": Colors.OKBLUE,
        "SUCCESS": Colors.OKGREEN,
        "WARNING": Colors.WARNING,
        "ERROR": Colors.FAIL,
        "DEBUG": Colors.HEADER
    }.get(level, Colors.OKBLUE)

    # Вывод в консоль с цветами
    console_msg = f"{color}[{timestamp}] [{level}]{Colors.ENDC} {message}"
    print(console_msg, file=sys.stderr if level in ["ERROR", "WARNING"] else sys.stdout)

    # Запись в лог-файл без цветов
    logging.log(
        getattr(logging, level, logging.INFO),
        f"[{level}] {message}"
    )


def validate_doi(doi: Any) -> bool:
    """Проверяет валидность DOI с улучшенной валидацией"""
    if not isinstance(doi, str) or not doi.strip():
        return False

    doi_patterns = [
        r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$',
        r'^https?://doi\.org/10\.\d{4,9}/[-._;()/:A-Z0-9]+$'
    ]

    return any(re.match(pattern, doi, re.I) for pattern in doi_patterns)


def normalize_doi(doi: str) -> str:
    """Нормализует DOI к стандартному формату"""
    if not doi:
        return ""

    doi = doi.strip().lower()
    if doi.startswith('http'):
        return doi
    return f'https://doi.org/{doi}'
