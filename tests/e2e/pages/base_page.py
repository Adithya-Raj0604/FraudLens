"""Base Page Object — shared navigation and wait helpers for all page objects."""

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

DEFAULT_TIMEOUT = 10


class BasePage:
    def __init__(self, driver, base_url: str):
        self.driver = driver
        self.base_url = base_url

    def open(self, path: str = "/"):
        self.driver.get(self.base_url.rstrip("/") + path)
        return self

    def wait_visible(self, locator, timeout: int = DEFAULT_TIMEOUT):
        return WebDriverWait(self.driver, timeout).until(EC.visibility_of_element_located(locator))

    def wait_present(self, locator, timeout: int = DEFAULT_TIMEOUT):
        return WebDriverWait(self.driver, timeout).until(EC.presence_of_element_located(locator))

    def find(self, locator):
        return self.driver.find_element(*locator)

    def find_all(self, locator):
        return self.driver.find_elements(*locator)
