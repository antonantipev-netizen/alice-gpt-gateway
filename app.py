import os
import re
import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_RESPONSES_URL = "https://api.groq.com/openai/v1/responses"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
VKUSVILL_MCP_URL = os.getenv("VKUSVILL_MCP_URL", "https://mcp.vkusvill.ru/mcp")

PERSONAL_PROFILE = (
    "Пользователя зовут Антон. Обращайся к нему на ты. "
    "Он предпочитает короткие, практичные ответы без лишней воды. "
    "Ты — его персональный голосовой ассистент Джарвис."
)

BASE_SYSTEM_PROMPT = (
    "Ты Джарвис, персональный голосовой ИИ-ассистент Антона. "
    "Отвечай по-русски, естественно, кратко и по существу. "
    "Это голосовой интерфейс: не используй Markdown, таблицы и длинные списки. "
    "Обычно отвечай в 2-4 предложениях. "
    "Если вопрос сложный, сначала дай короткий полезный ответ. "
    "Учитывай контекст текущего разговора и долговременную память, если она есть."
)

EXIT_WORDS = {
    "выход", "выйти", "стоп", "хватит", "закончить", "завершить",
    "закройся", "до свидания"
}

MEMORY_SHOW_PHRASES = {
    "что ты обо мне помнишь",
    "что ты помнишь обо мне",
    "что ты запомнил обо мне",
    "что ты запомнил",
    "что помнишь обо мне",
    "что помнишь"
}

MEMORY_CLEAR_PHRASES = {
    "забудь всё",
    "забудь все",
    "очисти память",
    "сотри память",
    "удали память"
}


def clean_for_voice(text: str) -> str:
    text = re.sub(r"[*_#>\x60]+", "", text or "")
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1000]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def strip_jarvis_prefix(command: str) -> str:
    normalized = normalize_text(command)
    low = normalized.lower()
    prefixes = [
        "джарвис ассистент, ",
        "джарвис ассистент ",
        "джарвис, ",
        "джарвис ",
    ]
    for prefix in prefixes:
        if low.startswith(prefix):
            return normalized[len(prefix):].strip()
    return normalized


def extract_remember_payload(command: str) -> str | None:
    normalized = strip_jarvis_prefix(command)
    low = normalized.lower()
    prefixes = [
        "запомни, что ",
        "запомни что ",
        "запомни, ",
        "запомни ",
    ]
    for prefix in prefixes:
        if low.startswith(prefix):
            payload = normalized[len(prefix):].strip(" .")
            return payload or None
    return None


def state_size(value: str) -> int:
    return len(("{\"memory\":\"" + value + "\"}").encode("utf-8"))


def compact_memory(current: str, new_fact: str) -> str:
    current = normalize_text(current)
    new_fact = normalize_text(new_fact)

    facts = []
    seen = set()

    for item in (current.split(" | ") if current else []) + [new_fact]:
        item = item.strip(" .|")
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(item)

    while facts and state_size(" | ".join(facts)) > 800:
        facts.pop(0)

    return " | ".join(facts)


def trim_history(history: list[dict], max_chars: int = 500) -> list[dict]:
    clean = []

    for item in history[-6:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        item_content = normalize_text(str(item.get("content", "")))
        if role not in {"user", "assistant"} or not item_content:
            continue
        clean.append({"role": role, "content": item_content[:160]})

    while clean and len(str(clean).encode("utf-8")) > max_chars:
        clean.pop(0)

    return clean[-4:]


def get_long_memory(data: dict) -> str:
    state = data.get("state", {})

    user_state = state.get("user", {})
    if isinstance(user_state, dict):
        memory = user_state.get("memory")
        if isinstance(memory, str) and memory.strip():
            return normalize_text(memory)

    app_state = state.get("application", {})
    if isinstance(app_state, dict):
        memory = app_state.get("memory")
        if isinstance(memory, str) and memory.strip():
            return normalize_text(memory)

    return ""


def get_session_history(data: dict) -> list[dict]:
    state = data.get("state", {})
    session_state = state.get("session", {})
    if not isinstance(session_state, dict):
        return []

    history = session_state.get("history", [])
    if not isinstance(history, list):
        return []

    return trim_history(history)


def build_system_prompt(memory: str) -> str:
    prompt = BASE_SYSTEM_PROMPT + " " + PERSONAL_PROFILE

    if memory:
        prompt += (
            " Долговременная память об Антоне: "
            + memory
            + ". Используй эти сведения только когда они уместны."
        )

    return prompt


def call_groq(user_text: str, history: list[dict], memory: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")

    messages = [{"role": "system", "content": build_system_prompt(memory)}]
    messages.extend(trim_history(history))
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



def is_vkusvill_intent(text: str) -> bool:
    low = normalize_text(text).lower()
    triggers = (
        "вкусвилл", "вкус вилл", "корзин", "закажи продукт",
        "купи продукт", "собери продукт", "закажи еду", "купи еду",
        "продукты домой", "доставка продуктов",
    )
    return any(trigger in low for trigger in triggers)


def extract_responses_text(payload: dict) -> str:
    parts = []
    for item in payload.get("output", []):
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                parts.append(content["text"])
    return normalize_text(" ".join(parts))


def call_vkusvill_agent(user_text: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")

    instructions = (
        "Ты Джарвис, помощник Антона по покупкам во ВкусВилле. "
        "Используй MCP ВкусВилла, чтобы найти подходящие товары и, когда запрос "
        "достаточно конкретный, создать ссылку на корзину. "
        "Не выдумывай товары, цены или наличие. "
        "Если есть несколько вариантов, предпочитай популярный и хорошо оцененный товар, "
        "если Антон не попросил дешевле, конкретный бренд, состав или КБЖУ. "
        "Корзина может содержать максимум 20 позиций. "
        "Не пытайся оформлять оплату: твоя задача — подготовить корзину и вернуть share_basket ссылку. "
        "Ответ по-русски, коротко. Если корзина создана, обязательно верни полную ссылку."
    )

    response = requests.post(
        GROQ_RESPONSES_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "instructions": instructions,
            "input": user_text,
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "vkusvill",
                    "server_url": VKUSVILL_MCP_URL,
                    "server_description": (
                        "Официальный MCP ВкусВилла: поиск товаров, детали товара "
                        "и создание ссылки на корзину."
                    ),
                    "require_approval": "never",
                    "allowed_tools": [
                        "vkusvill_products_search",
                        "vkusvill_product_details",
                        "vkusvill_cart_link_create",
                    ],
                }
            ],
        },
        timeout=18,
    )

    if not response.ok:
        raise RuntimeError(
            f"Groq Responses {response.status_code}: {response.text[:700]}"
        )

    answer = extract_responses_text(response.json())
    if not answer:
        raise RuntimeError("Groq Responses returned no final assistant text")

    return answer


def make_alice_response(
    version: str,
    session: dict,
    answer: str,
    history: list[dict],
    memory: str,
    end_session: bool = False,
    clear_memory: bool = False,
    tts_answer: str | None = None,
) -> dict:
    result = {
        "version": version,
        "session": session,
        "response": {
            "text": answer,
            "tts": tts_answer or answer,
            "end_session": end_session,
        },
        "session_state": {
            "history": trim_history(history),
        },
    }

    user_present = isinstance(session.get("user"), dict) and bool(
        session.get("user", {}).get("user_id")
    )

    if clear_memory:
        if user_present:
            result["user_state_update"] = {"memory": None}
        result["application_state"] = {}
    else:
        memory_state = {"memory": memory}
        if user_present:
            result["user_state_update"] = memory_state
        result["application_state"] = memory_state

    return result


@app.get("/")
def root():
    return jsonify(
        {
            "service": "Jarvis Alice Gateway",
            "status": "ok",
            "model": MODEL,
            "memory": "yandex-state",
        }
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "groq_configured": bool(GROQ_API_KEY),
            "model": MODEL,
            "memory_backend": "yandex-state",
        }
    )


@app.get("/vkusvill")
def vkusvill():
    text = request.args.get("q", "").strip()
    if not text:
        return jsonify({"error": "Use ?q=what to buy"}), 400

    try:
        answer = call_vkusvill_agent(text)
        return jsonify({"request": text, "answer": answer})
    except requests.Timeout:
        return jsonify({"error": "VkusVill agent timeout"}), 504
    except Exception as exc:
        print(f"VkusVill error: {exc}", flush=True)
        return jsonify({"error": str(exc)}), 500


@app.get("/ask")
def ask():
    text = request.args.get("text", "").strip()
    if not text:
        return jsonify({"error": "Use ?text=your question"}), 400

    try:
        answer = call_groq(text, [], "")
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

    command_for_memory = strip_jarvis_prefix(command)
    low = normalize_text(command_for_memory).lower()
    history = get_session_history(data)
    memory = get_long_memory(data)

    state = data.get("state", {})
    session_user = session.get("user", {}) if isinstance(session, dict) else {}
    app_obj = session.get("application", {}) if isinstance(session, dict) else {}

    print(
        "MEMDBG "
        f"new={session.get('new')} "
        f"user_present={bool(isinstance(session_user, dict) and session_user.get('user_id'))} "
        f"app_present={bool(isinstance(app_obj, dict) and app_obj.get('application_id'))} "
        f"state_user={bool(isinstance(state.get('user'), dict) and state.get('user'))} "
        f"state_app={bool(isinstance(state.get('application'), dict) and state.get('application'))} "
        f"memory_len={len(memory)} "
        f"command={command_for_memory[:80]!r}",
        flush=True,
    )

    if low in EXIT_WORDS:
        return jsonify(
            make_alice_response(
                version,
                session,
                "Хорошо, заканчиваю разговор.",
                history,
                memory,
                end_session=True,
            )
        )

    if low in MEMORY_CLEAR_PHRASES:
        return jsonify(
            make_alice_response(
                version,
                session,
                "Готово. Долговременную память очистил.",
                history,
                "",
                clear_memory=True,
            )
        )

    remember_payload = extract_remember_payload(command)
    if remember_payload:
        memory = compact_memory(memory, remember_payload)
        answer = f"Запомнил: {remember_payload}."
        history = history + [
            {"role": "user", "content": command_for_memory},
            {"role": "assistant", "content": answer},
        ]

        return jsonify(
            make_alice_response(
                version,
                session,
                answer,
                history,
                memory,
            )
        )

    if low in MEMORY_SHOW_PHRASES:
        if memory:
            answer = "Я помню: " + memory.replace(" | ", "; ") + "."
        else:
            answer = (
                "Пока в долговременной памяти ничего нет. "
                "Скажи: запомни, что я люблю футбол."
            )

        history = history + [
            {"role": "user", "content": command_for_memory},
            {"role": "assistant", "content": answer},
        ]

        return jsonify(
            make_alice_response(
                version,
                session,
                answer,
                history,
                memory,
            )
        )

    if not command_for_memory:
        answer = "Я на связи, Антон. Спрашивай."
        return jsonify(
            make_alice_response(
                version,
                session,
                answer,
                history,
                memory,
            )
        )

    try:
        if is_vkusvill_intent(command_for_memory):
            answer = call_vkusvill_agent(command_for_memory)
        else:
            answer = call_groq(command_for_memory, history, memory)

        history = history + [
            {"role": "user", "content": command_for_memory},
            {"role": "assistant", "content": answer},
        ]
    except requests.Timeout:
        answer = "Я не успел получить ответ. Спроси ещё раз покороче."
    except Exception as exc:
        print(f"AI error: {exc}", flush=True)
        answer = "Сейчас не получилось связаться с ИИ. Попробуй ещё раз."

    tts_answer = None
    if is_vkusvill_intent(command_for_memory) and "http" in answer:
        tts_answer = (
            "Готово. Я собрал корзину ВкусВилла. "
            "Ссылка есть в текстовом ответе."
        )

    return jsonify(
        make_alice_response(
            version,
            session,
            answer,
            history,
            memory,
            tts_answer=tts_answer,
        )
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
