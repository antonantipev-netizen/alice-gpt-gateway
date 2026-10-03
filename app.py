import os
import re
import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = (
    "Ты голосовой ИИ-ассистент Антона. "
    "Отвечай по-русски, естественно, кратко и по существу. "
    "Это голосовой интерфейс: не используй Markdown, таблицы и длинные списки. "
    "Обычно отвечай в 2-4 предложениях. "
    "Если вопрос сложный, сначала дай короткий полезный ответ. "
    "Учитывай предыдущие реплики текущего разговора."
)

EXIT_WORDS = {"выход", "выйти", "стоп", "хватит", "закончить", "завершить"}


def clean_for_voice(text: str) -> str:
    text = re.sub(r"[*_#>`]+", "", text or "")
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1000]


def call_groq(user_text: str, history: list[dict]) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": user_text})

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0.6,
            "reasoning_effort": "low",
            "reasoning_format": "hidden",
            "max_completion_tokens": 300,
        },
        timeout=3.2,
    )

    if not response.ok:
        raise RuntimeError(f"Groq {response.status_code}: {response.text[:500]}")

    answer = response.json()["choices"][0]["message"]["content"]
    return clean_for_voice(answer)


@app.get("/")
def root():
    return jsonify(
        {
            "service": "Alice GPT Gateway",
            "status": "ok",
            "model": MODEL,
        }
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "groq_configured": bool(GROQ_API_KEY),
            "model": MODEL,
        }
    )


@app.get("/ask")
def ask():
    text = request.args.get("text", "").strip()
    if not text:
        return jsonify({"error": "Use ?text=your question"}), 400

    try:
        answer = call_groq(text, [])
        return jsonify({"question": text, "answer": answer})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/alice")
def alice():
    data = request.get_json(silent=True) or {}
    version = data.get("version", "1.0")
    session = data.get("session", {})

    req = data.get("request", {})
    command = (
        req.get("original_utterance")
        or req.get("command")
        or ""
    ).strip()

    state = data.get("state", {}).get("session", {})
    history = state.get("history", [])
    if not isinstance(history, list):
        history = []

    if command.lower() in EXIT_WORDS:
        return jsonify(
            {
                "version": version,
                "session": session,
                "response": {
                    "text": "Хорошо, заканчиваю разговор.",
                    "tts": "Хорошо, заканчиваю разговор.",
                    "end_session": True,
                },
            }
        )

    if not command:
        answer = "Я на связи. Спрашивай."
        new_history = history
    else:
        try:
            answer = call_groq(command, history)
            new_history = (
                history
                + [
                    {"role": "user", "content": command},
                    {"role": "assistant", "content": answer},
                ]
            )[-8:]
        except requests.Timeout:
            answer = "Я не успел получить ответ. Спроси ещё раз покороче."
            new_history = history
        except Exception as exc:
            print(f"AI error: {exc}", flush=True)
            answer = "Сейчас не получилось связаться с ИИ. Попробуй ещё раз."
            new_history = history

    return jsonify(
        {
            "version": version,
            "session": session,
            "response": {
                "text": answer,
                "tts": answer,
                "end_session": False,
            },
            "session_state": {
                "history": new_history,
            },
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
