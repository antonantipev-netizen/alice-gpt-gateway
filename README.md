# Alice GPT Gateway

Webhook gateway for Yandex Alice using Groq.

## Endpoints

- `GET /health` — health check
- `GET /ask?text=Привет` — direct test
- `POST /alice` — Yandex Dialogs webhook

## Environment

- `GROQ_API_KEY` — required
- `GROQ_MODEL` — optional, defaults to `openai/gpt-oss-120b`

## Render

Build command:

```
pip install -r requirements.txt
```

Start command:

```
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 30
```
