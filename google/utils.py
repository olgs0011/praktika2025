from datetime import datetime
from typing import Any, Optional
import sys
import re


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def log_message(message: str, level: str = "INFO") -> None:
    """Улучшенное логирование с цветами и уровнями важности"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    color = {
        "INFO": Colors.OKBLUE,
        "SUCCESS": Colors.OKGREEN,
        "WARNING": Colors.WARNING,
        "ERROR": Colors.FAIL,
        "DEBUG": Colors.HEADER
    }.get(level.upper(), Colors.OKBLUE)

    print(f"{color}[{timestamp}] [{level.upper()}]{Colors.ENDC} {message}",
          file=sys.stderr if level.upper() in ["ERROR", "WARNING"] else sys.stdout)


def validate_doi(doi: Any) -> bool:
    """Проверяет валидность DOI"""
    if not isinstance(doi, str) or not doi.strip():
        return False
    return bool(re.match(r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$', doi, re.I))
