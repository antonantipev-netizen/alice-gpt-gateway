# Jarvis VkusVill Checkout Worker

Контейнер для Synology. Он получает задания от Jarvis Gateway и открывает share_basket ВкусВилла в Chromium.

## Текущий режим

Только DRY RUN / inspection:

- открывает только HTTPS-адреса на `vkusvill.ru`;
- сохраняет браузерный профиль в `./data/browser`;
- сохраняет последний скриншот в `./data/last_checkout.png`;
- определяет, требуется ли вход в аккаунт;
- отправляет на Gateway только безопасную диагностику интерфейса;
- НЕ нажимает кнопку подтверждения заказа;
- НЕ оформляет и НЕ оплачивает заказ.

## Переменные

- `GATEWAY_URL` — Jarvis Gateway на Render.
- `WORKER_TOKEN` — общий секрет worker ↔ Gateway.
- `HEADLESS=1` — Chromium без интерфейса.
- `POLL_SECONDS=5` — частота проверки очереди.

Запуск через Synology Container Manager будет настроен после того, как Gateway получит `CHECKOUT_WORKER_TOKEN`.
