import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


GATEWAY_URL = os.getenv(
    "GATEWAY_URL",
    "https://alice-gpt-gateway.onrender.com",
).rstrip("/")
WORKER_TOKEN = os.getenv("WORKER_TOKEN", "").strip()
POLL_SECONDS = max(3, int(os.getenv("POLL_SECONDS", "5")))
BROWSER_DATA_DIR = Path(os.getenv("BROWSER_DATA_DIR", "/data/browser"))
HEADLESS = os.getenv("HEADLESS", "1") != "0"
SCREENSHOT_PATH = Path(os.getenv("SCREENSHOT_PATH", "/data/last_checkout.png"))


def headers() -> dict:
    return {
        "Authorization": "Bearer " + WORKER_TOKEN,
        "Content-Type": "application/json",
        "User-Agent": "Jarvis-VkusVill-Checkout-Worker/1.0",
    }


def validate_vkusvill_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        host == "vkusvill.ru" or host.endswith(".vkusvill.ru")
    ):
        raise ValueError("blocked_non_vkusvill_url")
    return url


def next_task() -> dict | None:
    response = requests.get(
        GATEWAY_URL + "/vkusvill/checkout/worker/next",
        headers=headers(),
        timeout=20,
    )
    if response.status_code == 204:
        return None
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else None


def report(
    task_id: str,
    status: str,
    message: str = "",
    diagnostics: dict | None = None,
) -> None:
    payload = {
        "task_id": task_id,
        "status": status,
        "message": message[:300],
    }
    if diagnostics:
        payload["diagnostics"] = diagnostics

    response = requests.post(
        GATEWAY_URL + "/vkusvill/checkout/worker/report",
        headers=headers(),
        json=payload,
        timeout=20,
    )
    response.raise_for_status()


def clean_texts(values: list[str], limit: int) -> list[str]:
    result = []
    seen = set()
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text[:120])
        if len(result) >= limit:
            break
    return result


def collect_diagnostics(page) -> dict:
    buttons = []
    links = []
    inputs = []

    try:
        buttons = page.locator("button").all_inner_texts()
    except Exception:
        pass

    try:
        links = page.locator("a").all_inner_texts()
    except Exception:
        pass

    try:
        locators = page.locator("input")
        count = min(locators.count(), 20)
        for index in range(count):
            item = locators.nth(index)
            input_type = item.get_attribute("type") or "text"
            placeholder = item.get_attribute("placeholder") or ""
            label = f"{input_type}: {placeholder}".strip()
            inputs.append(label)
    except Exception:
        pass

    return {
        "title": page.title()[:160],
        "url": page.url[:500],
        "buttons": clean_texts(buttons, 30),
        "links": clean_texts(links, 30),
        "inputs": clean_texts(inputs, 20),
    }


def looks_like_login(diagnostics: dict) -> bool:
    url = str(diagnostics.get("url") or "").lower()
    if any(token in url for token in ("/login", "/auth", "signin", "sign-in")):
        return True

    text = " ".join(
        diagnostics.get("buttons", [])
        + diagnostics.get("links", [])
        + diagnostics.get("inputs", [])
    ).lower()

    login_markers = (
        "войти", "вход", "номер телефона", "телефон",
        "получить код", "код из смс", "sms",
    )
    return any(marker in text for marker in login_markers)


def inspect_cart(task: dict) -> None:
    task_id = str(task.get("task_id") or "")
    cart_url = validate_vkusvill_url(str(task.get("cart_url") or ""))

    report(task_id, "working", "Открываю корзину ВкусВилла.")

    BROWSER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(BROWSER_DATA_DIR),
            headless=HEADLESS,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
            viewport={"width": 1440, "height": 1000},
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(
                cart_url,
                wait_until="domcontentloaded",
                timeout=60_000,
            )
            try:
                page.wait_for_load_state("networkidle", timeout=15_000)
            except PlaywrightTimeoutError:
                pass
            page.wait_for_timeout(2500)

            diagnostics = collect_diagnostics(page)
            try:
                page.screenshot(
                    path=str(SCREENSHOT_PATH),
                    full_page=True,
                )
            except Exception:
                pass

            if looks_like_login(diagnostics):
                report(
                    task_id,
                    "needs_login",
                    "ВкусВилл просит авторизацию в браузерной сессии.",
                    diagnostics,
                )
            else:
                report(
                    task_id,
                    "inspected",
                    "Корзина открыта. DRY RUN: заказ не оформлялся.",
                    diagnostics,
                )
        finally:
            context.close()


def main() -> None:
    if not WORKER_TOKEN:
        raise RuntimeError("WORKER_TOKEN is required")

    print("Jarvis checkout worker started", flush=True)
    print(f"Gateway: {GATEWAY_URL}", flush=True)
    print(f"Headless: {HEADLESS}", flush=True)

    while True:
        try:
            task = next_task()
            if not task:
                time.sleep(POLL_SECONDS)
                continue

            task_id = str(task.get("task_id") or "")
            try:
                inspect_cart(task)
            except Exception as exc:
                print(
                    f"Checkout task failed: {type(exc).__name__}",
                    flush=True,
                )
                try:
                    report(
                        task_id,
                        "error",
                        type(exc).__name__,
                    )
                except Exception:
                    pass
        except requests.RequestException as exc:
            print(
                f"Gateway unavailable: {type(exc).__name__}",
                flush=True,
            )
            time.sleep(max(POLL_SECONDS, 10))
        except Exception as exc:
            print(
                f"Worker loop error: {type(exc).__name__}",
                flush=True,
            )
            time.sleep(max(POLL_SECONDS, 10))


if __name__ == "__main__":
    main()
