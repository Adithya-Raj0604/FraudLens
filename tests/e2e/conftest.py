"""
Shared fixtures for the Selenium UI test suite.

Points at a running frontend server and backend API — start both before
running these tests locally (see README's Test Plan section), or let the
GitHub Actions workflow start them for you.

Env vars:
    UI_BASE_URL   frontend URL (default http://localhost:5173, the Vite dev server)
    HEADLESS      "false" to watch the browser locally (default true)
"""

import os

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from tests.e2e.pages.dashboard_page import DashboardPage

UI_BASE_URL = os.environ.get("UI_BASE_URL", "http://localhost:5173")
HEADLESS = os.environ.get("HEADLESS", "true").lower() != "false"


@pytest.fixture
def driver():
    options = Options()
    if HEADLESS:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1440,1000")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    drv = webdriver.Chrome(options=options)
    yield drv
    drv.quit()


@pytest.fixture
def dashboard_page(driver):
    page = DashboardPage(driver, UI_BASE_URL)
    page.open()
    return page
