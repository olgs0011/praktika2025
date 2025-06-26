from typing import Dict, Optional
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
import time
import random
from utils import log_message


def get_scholar_data(doi: str) -> Optional[Dict]:
    """Парсинг через Selenium с человеческим поведением"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Для работы без GUI
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(
        f"user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )

        # Имитация человеческого поведения
        driver.get("https://scholar.google.com")
        time.sleep(random.uniform(2, 5))

        search_box = driver.find_element(By.NAME, "q")
        search_box.send_keys(doi)
        time.sleep(random.uniform(1, 2))
        search_box.submit()
        time.sleep(random.uniform(5, 10))  # Ожидание загрузки

        # Прокрутка страницы
        driver.execute_script("window.scrollBy(0, 500)")
        time.sleep(random.uniform(2, 3))

        # Сбор данных
        cited_by = []
        cites = []

        try:
            for item in driver.find_elements(By.CSS_SELECTOR, 'a.gs_or_cit[href*="cites"]'):
                cited_by.append(item.get_attribute("href"))
        except:
            pass

        try:
            for item in driver.find_elements(By.CSS_SELECTOR, 'a.gs_or_cit[href*="cluster"]'):
                cites.append(item.get_attribute("href"))
        except:
            pass

        return {
            'cited_by': cited_by[:5],
            'references': cites[:5]
        }

    except WebDriverException as e:
        log_message(f"Selenium ошибка: {str(e)}", "ERROR")
        return None
    finally:
        try:
            driver.quit()
        except:
            pass
