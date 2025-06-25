from datetime import datetime


def log_message(message: str, level: str = "INFO"):
    """Улучшенное логирование с уровнями важности"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}")