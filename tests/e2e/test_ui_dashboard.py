"""
UI test suite for the FraudLens dashboard (Page Object Model, Selenium + pytest).

Requires the frontend (Vite dev server) and backend (uvicorn) running — see
README's Test Plan section for how to start both locally, or run via CI where
the workflow starts them automatically.

Run: pytest tests/e2e -v
"""

import os

import pytest

from tests.e2e.pages.dashboard_page import DashboardPage

ANTHROPIC_KEY_PRESENT = bool(os.environ.get("ANTHROPIC_API_KEY"))


# ── Happy path ────────────────────────────────────────────────────────────────

def test_dashboard_loads_with_header_and_default_preset(dashboard_page):
    """Smoke test: the page renders its header and starts on the first preset."""
    header = dashboard_page.wait_visible(dashboard_page.HEADER_TITLE)
    assert "FraudLens" in header.text
    assert dashboard_page.active_preset_id() == "legit"
    assert dashboard_page.field_value("amount") == 1500


def test_preset_selection_populates_form_fields(dashboard_page):
    """Clicking a preset overwrites every form field with that preset's values."""
    dashboard_page.select_preset("drain")

    assert dashboard_page.active_preset_id() == "drain"
    assert dashboard_page.field_value("amount") == 9000
    assert dashboard_page.field_value("oldbalanceOrg") == 9000
    assert dashboard_page.field_value("newbalanceOrig") == 0
    assert dashboard_page.field_value("velocity-24hr") == 8
    assert dashboard_page.find(dashboard_page.TYPE_SELECT).get_attribute("value") == "TRANSFER"


def test_api_status_shows_connected_when_backend_reachable(dashboard_page):
    """With the backend up and the model loaded, the health indicator goes green."""
    dashboard_page.wait_for_api_status("online")


def test_submit_investigation_reaches_terminal_state(dashboard_page):
    """
    Submitting a transaction shows the live investigation feed and eventually
    settles into a terminal state (a finished report, or a graceful error if
    the agent can't run — e.g. no ANTHROPIC_API_KEY configured), re-enabling
    the submit button either way.
    """
    dashboard_page.select_preset("drain")
    dashboard_page.submit()

    dashboard_page.wait_visible(dashboard_page.INVESTIGATION_FEED)
    # 60s: if a real ANTHROPIC_API_KEY is configured (local dev), this hits the
    # live agent — RAG retrieval + Claude latency needs headroom beyond the
    # near-instant error path CI takes by default (no key configured).
    terminal_status = dashboard_page.wait_for_terminal_investigation(timeout=60)

    assert terminal_status in ("done", "error")
    assert dashboard_page.is_submit_disabled() is False


@pytest.mark.skipif(
    not ANTHROPIC_KEY_PRESENT,
    reason="ANTHROPIC_API_KEY not set — skipping live-agent E2E test",
)
def test_full_investigation_renders_report_and_shap_chart(dashboard_page):
    """With a real Anthropic key configured, a full investigation produces a report + SHAP chart."""
    dashboard_page.select_preset("drain")
    dashboard_page.submit()

    status = dashboard_page.wait_for_terminal_investigation(timeout=60)
    assert status == "done"

    report = dashboard_page.wait_visible(dashboard_page.INVESTIGATION_REPORT)
    assert len(report.text) > 50
    dashboard_page.wait_visible(dashboard_page.SHAP_CHART)


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_manual_edit_clears_active_preset_highlight(dashboard_page):
    """Editing a field by hand deselects whichever preset was active."""
    dashboard_page.select_preset("mule")
    assert dashboard_page.active_preset_id() == "mule"

    dashboard_page.set_field("amount", 12345)

    assert dashboard_page.active_preset_id() is None
    assert dashboard_page.field_value("amount") == 12345


def test_amount_field_clamps_to_minimum(dashboard_page):
    """Decrementing below the field's min (0) clamps instead of going negative."""
    dashboard_page.set_field("amount", 50)  # step is 100, so one decrement would go negative
    dashboard_page.click_decrement("amount")

    assert dashboard_page.field_value("amount") == 0

    dashboard_page.click_decrement("amount")  # already at the floor
    assert dashboard_page.field_value("amount") == 0


def test_api_status_shows_offline_when_backend_unreachable(driver):
    """If /health is unreachable, the indicator reports offline instead of hanging on 'checking'."""
    driver.execute_cdp_cmd("Network.enable", {})
    driver.execute_cdp_cmd("Network.setBlockedURLs", {"urls": ["*/health*"]})

    page = DashboardPage(driver, os.environ.get("UI_BASE_URL", "http://localhost:5173"))
    page.open()

    page.wait_for_api_status("offline")
