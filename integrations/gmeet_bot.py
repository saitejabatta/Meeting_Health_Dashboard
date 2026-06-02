# integrations/gmeet_bot.py
"""Google Meet browser automation helper with clear fallback behavior."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from integrations.audio_router import AudioRouter


@dataclass
class JoinResult:
    """Result of attempting to join a Google Meet."""

    joined: bool
    message: str


class GoogleMeetBot:
    """Uses Selenium/undetected-chromedriver to open Google Meet when configured."""

    def __init__(self) -> None:
        """Load Google credentials and audio routing configuration."""
        if load_dotenv is not None:
            load_dotenv()
        self.email = os.getenv("GOOGLE_EMAIL", "")
        self.password = os.getenv("GOOGLE_PASSWORD", "")
        self.audio_router = AudioRouter()
        self.driver = None

    def join(self, meet_url: str, display_name: str = "") -> JoinResult:
        """Open a Meet URL and attempt to click through the join screen."""
        routing = self.audio_router.check()
        if not routing.ok:
            return JoinResult(False, routing.message)
        try:
            import undetected_chromedriver as uc
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support import expected_conditions as expected
            from selenium.webdriver.support.ui import WebDriverWait
        except ImportError:
            return JoinResult(False, self.manual_join_instructions(meet_url, "Selenium or undetected-chromedriver is not installed.", display_name=display_name))
        try:
            options = uc.ChromeOptions()
            options.add_argument("--use-fake-ui-for-media-stream")
            options.add_argument("--disable-notifications")
            self.driver = uc.Chrome(options=options)
            self.driver.get(meet_url)
            wait = WebDriverWait(self.driver, 20)
            if display_name:
                for selector in [
                    "//input[contains(translate(@aria-label, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'name')]",
                    "//input[contains(translate(@placeholder, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'name')]",
                    "//input[@type='text']",
                ]:
                    with _suppress_webdriver_errors():
                        name_input = self.driver.find_element(By.XPATH, selector)
                        name_input.clear()
                        name_input.send_keys(display_name)
                        break
            buttons = self.driver.find_elements(By.TAG_NAME, "button")
            for button in buttons:
                label = (button.get_attribute("aria-label") or button.text or "").lower()
                if "microphone" in label or "mic" in label:
                    with _suppress_webdriver_errors():
                        button.click()
            join_button = wait.until(
                expected.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Join now') or contains(text(), 'Ask to join')]/ancestor::button"))
            )
            join_button.click()
            name_note = f" Requested display name '{display_name}'." if display_name else ""
            return JoinResult(True, f"Opened Google Meet and attempted to join.{name_note} Route audio through {routing.device_name}.")
        except Exception as exc:
            return JoinResult(False, self.manual_join_instructions(meet_url, str(exc), display_name=display_name))

    def leave(self) -> JoinResult:
        """Leave the Meet if a browser driver is active."""
        if self.driver is not None:
            with _suppress_webdriver_errors():
                self.driver.quit()
        return JoinResult(True, "Google Meet browser closed.")

    def manual_join_instructions(self, meet_url: str, reason: str = "", display_name: str = "") -> str:
        """Return manual Google Meet joining instructions."""
        reason_text = f" Automatic join failed: {reason}" if reason else ""
        name_text = f" Use display name '{display_name}' so participants know an assistant is attending." if display_name else ""
        return (
            f"Open {meet_url} manually and join the meeting.{reason_text}{name_text} "
            f"Set browser output to your virtual audio route and keep VIRTUAL_AUDIO_DEVICE_NAME={self.audio_router.device_name}."
        )


class _suppress_webdriver_errors:
    """Context manager that ignores best-effort browser automation failures."""

    def __enter__(self) -> "_suppress_webdriver_errors":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return True
