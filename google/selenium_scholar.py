from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from urllib.parse import quote
import time
import random
from typing import Dict, Optional
from utils import log_message
from selenium.webdriver.common.keys import Keys
import re

def get_scholar_data(search_query: str, is_doi: bool = True) -> Optional[Dict]:
    """Поиск данных в Google Scholar с расширенной обработкой капчи"""
    # Настройка браузера для имитации человеческого поведения
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-extensions")

    driver = webdriver.Chrome(options=chrome_options)
    try:
        # Установка увеличенного времени ожидания загрузки страницы
        driver.set_page_load_timeout(120)
        
        # Стратегия поиска по DOI
        if is_doi:
            clean_doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', search_query)
            search_url = f"https://scholar.google.com/scholar?hl=en&q={quote(clean_doi)}"
            log_message(f"Загрузка страницы: {search_url}", "DEBUG")
            
            driver.get(search_url)
            time.sleep(random.uniform(5, 8))  # Увеличенная задержка

            # Обработка капчи (3 попытки с увеличением времени)
            for attempt in range(3):
                if check_captcha(driver):
                    handle_captcha(driver, attempt+1)
                    time.sleep(10)  # Пауза после ввода капчи
                    driver.refresh()
                    time.sleep(random.uniform(3, 5))
                else:
                    break

            # Проверка наличия статьи
            if f"DOI:{clean_doi}" in driver.page_source:
                log_message("Статья найдена по DOI", "DEBUG")
                return extract_citations(driver)

        # Альтернативная стратегия поиска через главную страницу
        log_message("Попытка альтернативного поиска", "DEBUG")
        driver.get("https://scholar.google.com/")
        time.sleep(random.uniform(4, 7))

        # Проверка капчи на главной странице
        if check_captcha(driver):
            handle_captcha(driver, extended_time=True)

        # Выполнение поискового запроса
        search_box = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.NAME, "q"))
        )
        search_box.clear()
        
        query = f'doi:{search_query}' if is_doi else search_query
        search_box.send_keys(query)
        time.sleep(random.uniform(1, 3))  # Имитация человеческого ввода
        
        search_box.send_keys(Keys.RETURN)
        time.sleep(random.uniform(6, 9))  # Увеличенное время ожидания

        # Финальная проверка капчи
        if check_captcha(driver):
            handle_captcha(driver, extended_time=True)

        return extract_citations(driver)

    except TimeoutException:
        log_message("Превышено время ожидания загрузки страницы", "ERROR")
        return None
    except Exception as e:
        log_message(f"Критическая ошибка: {str(e)}", "ERROR")
        return None
    finally:
        driver.quit()

def check_captcha(driver) -> bool:
    """Проверка наличия капчи с расширенной детекцией"""
    try:
        captcha_indicators = [
            "CAPTCHA", "докажите что вы не робот", "recaptcha",
            "sorry", "подтвердите", "verification", "robot", "captcha"
        ]
        page_text = driver.page_source.lower()
        return any(indicator.lower() in page_text for indicator in captcha_indicators)
    except Exception:
        return False

def handle_captcha(driver, attempt: int = 1, extended_time: bool = False):
    """Расширенная обработка капчи с ручным вводом"""
    if attempt > 1:
        log_message(f"Попытка {attempt} решения капчи", "WARNING")
    
    if extended_time:
        log_message("У вас есть 5 минут для решения капчи...", "INFO")
        print("="*50)
        print("ПОЯВИЛАСЬ КАПЧА! ПОЖАЛУЙСТА:")
        print("1. Решите капчу в открывшемся браузере")
        print("2. После успешного решения вернитесь сюда")
        print("3. Нажмите Enter чтобы продолжить")
        print("="*50)
        
        # Открываем капчу в новом окне для удобства
        driver.execute_script("window.open('https://scholar.google.com')")
        driver.switch_to.window(driver.window_handles[-1])
        
        input("Нажмите Enter после решения капчи... ")
        time.sleep(5)  # Дополнительная пауза
    else:
        log_message("Обнаружена капча. Пожалуйста, решите её в браузере...", "WARNING")
        input("Нажмите Enter после решения капчи... ")
        time.sleep(3)

def extract_citations(driver) -> Dict:
    """Извлечение данных о цитированиях с увеличенными таймаутами"""
    try:
        cited_by = []
        references = []
        
        # Ожидание загрузки основного контента (увеличенный таймаут)
        WebDriverWait(driver, 25).until(
            EC.presence_of_element_located((By.ID, "gs_res_ccl_mid"))
        )

        # Обработка "Cited by"
        try:
            cited_by_link = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, '//a[contains(text(), "Cited by")]'))
            )
            cited_by_count = int(cited_by_link.text.replace("Cited by", "").strip())
            log_message(f"Найдено цитирований: {cited_by_count}", "DEBUG")

            if cited_by_count > 0:
                cited_by_link.click()
                time.sleep(random.uniform(4, 7))  # Увеличенная задержка

                # Сбор ссылок на цитирующие статьи
                cited_by_elements = WebDriverWait(driver, 20).until(
                    EC.presence_of_all_elements_located((By.XPATH, '//a[contains(@href, "doi.org")]'))
                )
                cited_by = [el.get_attribute("href") for el in cited_by_elements[:5]]
                
                # Возврат на исходную страницу
                driver.back()
                time.sleep(random.uniform(3, 5))
                
                # Ожидание возврата
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.ID, "gs_res_ccl_mid"))
                )

        except TimeoutException:
            log_message("Таймаут при обработке 'Cited by'", "DEBUG")
        except Exception as e:
            log_message(f"Ошибка при обработке цитирований: {str(e)}", "DEBUG")

        # Обработка "References"
        try:
            references_link = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, '//a[contains(text(), "References")]'))
            )
            references_link.click()
            time.sleep(random.uniform(4, 7))

            # Сбор ссылок на статьи в списке литературы
            references_elements = WebDriverWait(driver, 20).until(
                EC.presence_of_all_elements_located((By.XPATH, '//a[contains(@href, "doi.org")]'))
            )
            references = [el.get_attribute("href") for el in references_elements[:5]]

        except TimeoutException:
            log_message("Таймаут при обработке 'References'", "DEBUG")
        except Exception as e:
            log_message(f"Ошибка при обработке ссылок: {str(e)}", "DEBUG")

        log_message(f"Результат: {len(cited_by)} цитирований, {len(references)} ссылок", "INFO")
        return {
            'cited_by': cited_by,
            'references': references
        }

    except Exception as e:
        log_message(f"Ошибка извлечения данных: {str(e)}", "ERROR")
        return None
