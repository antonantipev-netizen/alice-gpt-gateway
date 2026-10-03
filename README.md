# Jarvis Alice Gateway

Webhook gateway for Yandex Alice using Groq.

## Features

- Voice chat through Yandex Alice
- Short-term conversation context inside an Alice session
- Long-term memory between sessions through Yandex skill state
- Explicit memory commands:
  - `Запомни, что ...`
  - `Что ты обо мне помнишь?`
  - `Очисти память`
- Personal voice style for Anton

## Endpoints

- `GET /health` — health check
- `GET /ask?text=Привет` — direct Groq test
- `POST /alice` — Yandex Dialogs webhook

## Environment

- `GROQ_API_KEY` — required
- `GROQ_MODEL` — optional, defaults to `openai/gpt-oss-120b`

## Yandex Dialogs

Enable skill storage in the skill settings:

`Основные настройки → Хранилище → Использовать хранилище данных в навыке`

The skill uses:

- `session_state` for short-term context
- `user_state_update` for authorized-user memory across sessions/devices
- `application_state` as per-device fallback

## Render

Build command:

```
pip install -r requirements.txt
```

Start command:

```
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 30
```
