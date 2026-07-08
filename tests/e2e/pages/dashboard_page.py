"""Page Object for the FraudLens single-page dashboard (frontend/src/App.tsx)."""

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

from tests.e2e.pages.base_page import BasePage


class DashboardPage(BasePage):
    HEADER_TITLE = (By.XPATH, "//span[contains(text(), 'FraudLens')]")
    API_STATUS = (By.CSS_SELECTOR, "[data-testid='api-status']")
    TYPE_SELECT = (By.CSS_SELECTOR, "[data-testid='field-type']")
    SUBMIT_BUTTON = (By.CSS_SELECTOR, "[data-testid='submit-investigate']")
    INVESTIGATION_FEED = (By.CSS_SELECTOR, "[data-testid='investigation-feed']")
    INVESTIGATION_REPORT = (By.CSS_SELECTOR, "[data-testid='investigation-report']")
    SHAP_CHART = (By.CSS_SELECTOR, "[data-testid='shap-chart']")
    ACTIVE_PRESET = (By.CSS_SELECTOR, "[data-testid^='preset-'].border-white\\/30")

    def preset_button(self, preset_id: str):
        return (By.CSS_SELECTOR, f"[data-testid='preset-{preset_id}']")

    def field_input(self, field: str):
        return (By.CSS_SELECTOR, f"[data-testid='field-{field}-input']")

    def field_decrement(self, field: str):
        return (By.CSS_SELECTOR, f"[data-testid='field-{field}-decrement']")

    # ── Actions ──────────────────────────────────────────────────────────────

    def select_preset(self, preset_id: str):
        self.wait_visible(self.preset_button(preset_id)).click()
        return self

    def field_value(self, field: str) -> float:
        return float(self.find(self.field_input(field)).get_attribute("value"))

    def set_field(self, field: str, value) -> "DashboardPage":
        el = self.wait_visible(self.field_input(field))
        el.clear()
        el.send_keys(str(value))
        el.send_keys("\t")  # blur to commit the onChange
        return self

    def click_decrement(self, field: str, times: int = 1) -> "DashboardPage":
        btn = self.wait_visible(self.field_decrement(field))
        for _ in range(times):
            btn.click()
        return self

    def submit(self) -> "DashboardPage":
        self.wait_visible(self.SUBMIT_BUTTON).click()
        return self

    def is_submit_disabled(self) -> bool:
        return self.find(self.SUBMIT_BUTTON).get_attribute("disabled") is not None

    def active_preset_id(self) -> str | None:
        presets = self.find_all(self.ACTIVE_PRESET)
        if not presets:
            return None
        return presets[0].get_attribute("data-testid").removeprefix("preset-")

    def api_status_state(self) -> str:
        return self.wait_present(self.API_STATUS).get_attribute("data-status")

    def investigation_status(self) -> str:
        return self.wait_visible(self.INVESTIGATION_FEED).get_attribute("data-status")

    # ── Waits ────────────────────────────────────────────────────────────────

    def wait_for_api_status(self, expected: str, timeout: int = 10) -> "DashboardPage":
        WebDriverWait(self.driver, timeout).until(lambda d: self.api_status_state() == expected)
        return self

    def wait_for_terminal_investigation(self, timeout: int = 30) -> str:
        """Waits until the investigation feed leaves 'running' (i.e. reaches 'done' or 'error')."""
        WebDriverWait(self.driver, timeout).until(
            lambda d: self.investigation_status() in ("done", "error")
        )
        return self.investigation_status()
