from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import random
from typing import Dict, Optional
from utils import log_message


def human_like_typing(element, text):
    """Имитация человеческого ввода"""
    for char in text:
        element.send_keys(char)
        time.sleep(random.uniform(0.1, 0.3))


def handle_captcha_manually(driver):
    """Остановка для ручного ввода капчи"""
    if "докажите что вы не робот" in driver.page_source.lower():
        log_message("Обнаружена капча. Требуется ручной ввод!", "WARNING")
        print("\n=== ВНИМАНИЕ ===")
        print("1. Откройте браузер Chrome который появился")
        print("2. Решите капчу")
        print("3. После успешного ввода вернитесь сюда")
        print("4. Нажмите Enter в этом окне чтобы продолжить...")
        input()
        return True
    return False


def get_scholar_data(doi: str) -> Optional[Dict]:
    """Получение данных из Google Scholar с защитой от блокировок"""
    chrome_options = Options()

    # Настройки для уменьшения блокировок
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--window-size=1200,800")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    # Для отладки (показываем браузер)
    chrome_options.add_argument("--headless=new")  # Убрать для отладки

    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )

        # Первый запрос к главной странице
        driver.get("https://scholar.google.com")
        time.sleep(random.uniform(3, 5))

        # Проверка на капчу
        if handle_captcha_manually(driver):
            # Повторяем запрос после капчи
            driver.get("https://scholar.google.com")
            time.sleep(random.uniform(2, 4))

        # Ввод поискового запроса
        search_box = driver.find_element(By.NAME, "q")
        human_like_typing(search_box, doi)
        time.sleep(random.uniform(1, 2))
        search_box.submit()
        time.sleep(random.uniform(5, 8))

        # Проверка на капчу после поиска
        if handle_captcha_manually(driver):
            # Повторный поиск
            search_box = driver.find_element(By.NAME, "q")
            human_like_typing(search_box, doi)
            search_box.submit()
            time.sleep(random.uniform(5, 8))

        # Сохраняем страницу для отладки
        with open("last_page.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)

        # Сбор результатов
        cited_by = []
        cites = []

        # Пробуем разные варианты селекторов
        try:
            cited_elements = driver.find_elements(By.CSS_SELECTOR, 'a[href*="cites"]')
            cited_by = [el.get_attribute("href") for el in cited_elements[:3]]
        except:
            pass

        try:
            ref_elements = driver.find_elements(By.CSS_SELECTOR, 'a[href*="cluster"]')
            cites = [el.get_attribute("href") for el in ref_elements[:3]]
        except:
            pass

        return {
            'cited_by': cited_by,
            'references': cites
        }

    except Exception as e:
        log_message(f"Ошибка при парсинге: {str(e)}", "ERROR")
        return None
    finally:
        try:
            driver.quit()
        except:
            pass
