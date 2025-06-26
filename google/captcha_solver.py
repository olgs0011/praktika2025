from twocaptcha import TwoCaptcha
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import time
from dotenv import load_dotenv

load_dotenv()


class CaptchaSolver:
    def __init__(self, driver):
        self.driver = driver
        self.api_key = os.getenv('2CAPTCHA_API_KEY')

    def solve_recaptcha(self):
        """Решение reCAPTCHA через 2captcha"""
        solver = TwoCaptcha(self.api_key)

        # Получаем ключ сайта
        site_key = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.g-recaptcha'))
        ).get_attribute('data-sitekey')

        # Решаем капчу
        try:
            result = solver.recaptcha(
                sitekey=site_key,
                url=self.driver.current_url
            )

            # Вводим решение
            self.driver.execute_script(
                f"document.getElementById('g-recaptcha-response').innerHTML='{result['code']}';"
            )
            time.sleep(2)
            return True
        except Exception as e:
            print(f"Ошибка решения капчи: {e}")
            return False
