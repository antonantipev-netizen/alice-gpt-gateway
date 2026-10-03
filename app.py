import os
import re
import json
from pathlib import Path
from threading import Lock, Thread

import requests
from flask import Flask, jsonify, request
from vkusvill_direct import build_cart_direct_sync

app = Flask(__name__)

VKUSVILL_JOBS = {}
VKUSVILL_JOBS_LOCK = Lock()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_RESPONSES_URL = "https://api.groq.com/openai/v1/responses"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
SEARCH_MODEL = os.getenv("GROQ_SEARCH_MODEL", "openai/gpt-oss-20b")
VKUSVILL_MCP_URL = os.getenv("VKUSVILL_MCP_URL", "https://mcp.vkusvill.ru/mcp")
BITRIX_TASKS_WEBHOOK_URL = (
    os.getenv("BITRIX_TASKS_WEBHOOK_URL")
    or os.getenv("BITRIX_WEBHOOK_URL")
    or ""
).strip()
BITRIX_REQUEST_TAG = os.getenv("BITRIX_REQUEST_TAG", "SMR_Request").strip()

WORK_CONTEXT_PATH = Path(os.getenv("WORK_CONTEXT_PATH", "work_context.json"))


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



def extract_forget_payload(command: str) -> str | None:
    normalized = strip_jarvis_prefix(command)
    low = normalized.lower()
    prefixes = [
        "забудь про ",
        "забудь, что ",
        "забудь что ",
        "удали из памяти ",
        "не помни ",
    ]
    for prefix in prefixes:
        if low.startswith(prefix):
            payload = normalized[len(prefix):].strip(" .")
            return payload or None
    return None


def infer_auto_memory(command: str) -> str | None:
    """Save only stable, non-sensitive facts stated by the user."""
    text = normalize_text(strip_jarvis_prefix(command)).strip(" .")
    low = text.lower()

    if not text or len(text) > 180:
        return None

    # Questions and explicit opt-out are never auto-saved.
    question_starters = (
        "кто ", "что ", "где ", "когда ", "как ", "почему ", "зачем ",
        "можно ", "нужно ", "сколько ", "какой ", "какая ", "какие ",
    )
    if "?" in text or low.startswith(question_starters):
        return None
    if "не запоминай" in low or "не надо запоминать" in low:
        return None

    # Do not automatically persist secrets, health, exact location, finance, or other sensitive details.
    sensitive_markers = (
        "пароль", "пин", "pin", "код из смс", "cvv", "cvc", "номер карты",
        "паспорт", "снилс", "инн", "адрес", "улица", "квартира",
        "диагноз", "болит", "болезн", "лекар", "давление", "вес ", "рост ",
        "зарплат", "доход", "долг", "кредит", "счет ", "счёт ",
    )
    if any(marker in low for marker in sensitive_markers):
        return None

    transient_markers = (
        "сегодня", "завтра", "вчера", "сейчас", "на этой неделе",
        "временно", "сегодня вечером", "сегодня утром",
    )
    if any(marker in low for marker in transient_markers):
        return None

    patterns = [
        (r"^я люблю\s+(.+)$", "Любит: {}"),
        (r"^мне нравится\s+(.+)$", "Нравится: {}"),
        (r"^я предпочитаю\s+(.+)$", "Предпочитает: {}"),
        (r"^я не люблю\s+(.+)$", "Не любит: {}"),
        (r"^мой любимый\s+(.+)$", "Любимое: {}"),
        (r"^моя любимая\s+(.+)$", "Любимое: {}"),
        (r"^мои любимые\s+(.+)$", "Любимое: {}"),
        (r"^я работаю\s+(.+)$", "Работа: {}"),
        (r"^у меня машина\s+(.+)$", "Машина: {}"),
        (r"^у меня автомобиль\s+(.+)$", "Машина: {}"),
        (r"^у меня телефон\s+(.+)$", "Телефон: {}"),
        (r"^у меня ноутбук\s+(.+)$", "Ноутбук: {}"),
    ]

    for pattern, template in patterns:
        match = re.match(pattern, low, flags=re.IGNORECASE)
        if match:
            value = text[match.start(1):match.end(1)].strip(" .")
            if 2 <= len(value) <= 120:
                return template.format(value)

    return None


def forget_from_memory(current: str, needle: str) -> tuple[str, bool]:
    needle_low = normalize_text(needle).lower()
    facts = [x.strip() for x in current.split(" | ") if x.strip()]
    kept = [x for x in facts if needle_low not in x.lower()]
    return " | ".join(kept), len(kept) != len(facts)


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



BITRIX_STATUS_LABELS = {
    "1": "Новая",
    "2": "Ожидает выполнения",
    "3": "Выполняется",
    "4": "Ждёт контроля",
    "5": "Завершена",
    "6": "Отложена",
    "7": "Отклонена",
}


def bitrix_configured() -> bool:
    return bool(BITRIX_TASKS_WEBHOOK_URL)


def bitrix_base_url() -> str:
    base = BITRIX_TASKS_WEBHOOK_URL.rstrip("/") + "/"
    if not re.match(r"^https://[^/]+/rest/\d+/[^/]+/$", base):
        raise RuntimeError("BITRIX_URL_FORMAT")
    return base


def bitrix_api(method: str, params: dict | None = None) -> dict:
    allowed = {"tasks.task.list", "tasks.task.get"}
    if method not in allowed:
        raise RuntimeError("BITRIX_METHOD_FORBIDDEN")
    if not bitrix_configured():
        raise RuntimeError("BITRIX_NOT_CONFIGURED")

    url = bitrix_base_url() + method + ".json"
    response = requests.post(
        url,
        json=params or {},
        headers={"Content-Type": "application/json"},
        timeout=4.0,
        allow_redirects=False,
    )

    if 300 <= response.status_code < 400:
        location = response.headers.get("Location")
        if not location or not location.startswith(bitrix_base_url().split("/rest/")[0]):
            raise RuntimeError("BITRIX_REDIRECT_FORBIDDEN")
        response = requests.post(
            location,
            json=params or {},
            headers={"Content-Type": "application/json"},
            timeout=4.0,
            allow_redirects=False,
        )

    if not response.ok:
        raise RuntimeError(f"BITRIX_HTTP_{response.status_code}")

    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(f"BITRIX_{payload.get('error')}")
    return payload


def clean_bitrix_description(value: str) -> str:
    text = value or ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(?:p|div|li)>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[/?(?:B|I|U|LIST|\*|QUOTE)(?:=[^\]]*)?\]", "", text, flags=re.I)
    text = re.sub(r"\[URL=[^\]]+\]([\s\S]*?)\[/URL\]", r"\1", text, flags=re.I)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return normalize_text(text)[:700]


def normalize_bitrix_task(task: dict) -> dict:
    status = str(task.get("status") or task.get("STATUS") or "")
    tags = task.get("tags") or task.get("TAGS") or []
    if isinstance(tags, dict):
        tags = list(tags.values())
    clean_tags = []
    for item in tags or []:
        if isinstance(item, str):
            clean_tags.append(item)
        elif isinstance(item, dict):
            clean_tags.append(item.get("title") or item.get("name") or "")

    return {
        "id": int(task.get("id") or task.get("ID") or 0),
        "title": normalize_text(str(task.get("title") or task.get("TITLE") or ""))[:220],
        "description": clean_bitrix_description(str(task.get("description") or task.get("DESCRIPTION") or "")),
        "group_id": task.get("groupId") or task.get("GROUP_ID"),
        "responsible_id": task.get("responsibleId") or task.get("RESPONSIBLE_ID"),
        "created_date": task.get("createdDate") or task.get("CREATED_DATE"),
        "changed_date": task.get("changedDate") or task.get("CHANGED_DATE"),
        "deadline": task.get("deadline") or task.get("DEADLINE"),
        "status": status,
        "status_label": BITRIX_STATUS_LABELS.get(status, status or "—"),
        "tags": [x for x in clean_tags if x],
    }


def get_bitrix_requests(limit: int = 50) -> list[dict]:
    limit = max(1, min(limit, 100))
    select = [
        "ID", "TITLE", "DESCRIPTION", "GROUP_ID", "TAGS",
        "CREATED_BY", "RESPONSIBLE_ID", "CREATED_DATE",
        "CHANGED_DATE", "DEADLINE", "STATUS",
    ]
    tasks = []
    start = 0

    while len(tasks) < limit:
        payload = bitrix_api(
            "tasks.task.list",
            {
                "filter": {"TAG": BITRIX_REQUEST_TAG},
                "order": {"CHANGED_DATE": "desc", "ID": "desc"},
                "select": select,
                "start": start,
            },
        )
        result = payload.get("result") or {}
        page = result.get("tasks") or []
        if not isinstance(page, list):
            raise RuntimeError("BITRIX_TASK_LIST_FORMAT")

        tasks.extend(normalize_bitrix_task(task) for task in page)
        next_start = payload.get("next")
        if next_start is None:
            next_start = result.get("next")
        if next_start is None or len(tasks) >= limit:
            break
        start = int(next_start)

    return tasks[:limit]


def is_bitrix_intent(text: str) -> bool:
    low = normalize_text(text).lower()
    triggers = (
        "битрикс", "bitrix", "заявк", "задач смр",
        "просроч", "приостановлен", "отложен",
        "что в работе", "что назначено", "что закрыто",
        "какие заявки", "активные заявки", "заявки на сегодня",
    )
    return any(trigger in low for trigger in triggers)


def call_bitrix_agent(user_text: str, history: list[dict]) -> str:
    if not bitrix_configured():
        return (
            "Bitrix24 ещё не подключён к Джарвису. "
            "Нужно один раз добавить входящий webhook в Render."
        )

    tasks = get_bitrix_requests(50)
    if not tasks:
        return "В Bitrix24 не нашёл заявок с тегом SMR Request."

    history_text = []
    for item in trim_history(history):
        role = "Антон" if item.get("role") == "user" else "Джарвис"
        history_text.append(f"{role}: {item.get('content', '')}")

    prompt = (
        "Ты Джарвис, рабочий голосовой ассистент Антона. "
        "Ниже живые данные заявок СМР из Bitrix24. "
        "Отвечай ТОЛЬКО по этим данным. Не выдумывай статусы, сроки и объёмы. "
        "Если нужной заявки нет среди переданных данных, так и скажи. "
        "Статусы Bitrix: 1 новая, 2 ожидает выполнения, 3 выполняется, "
        "4 ждёт контроля, 5 завершена, 6 отложена, 7 отклонена. "
        "Ответ должен быть коротким и удобным для озвучивания: 2-5 предложений. "
        "Если перечисляешь заявки, называй сначала объект/название и статус.\n\n"
        "ЖИВЫЕ ЗАЯВКИ BITRIX24:\n"
        + json.dumps(tasks, ensure_ascii=False)
        + "\n\n"
    )
    if history_text:
        prompt += "КОНТЕКСТ ДИАЛОГА:\n" + " | ".join(history_text) + "\n\n"
    prompt += "ВОПРОС АНТОНА:\n" + user_text

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты анализируешь только переданные живые данные Bitrix24.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.15,
            "reasoning_effort": "low",
            "reasoning_format": "hidden",
            "max_completion_tokens": 280,
        },
        timeout=3.0,
    )

    if not response.ok:
        raise RuntimeError(f"Bitrix agent {response.status_code}: {response.text[:500]}")

    answer = response.json()["choices"][0]["message"]["content"]
    return clean_for_voice(answer)


def load_work_context() -> dict:
    try:
        with WORK_CONTEXT_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Work context load error: {exc}", flush=True)
        return {}


def is_work_intent(text: str) -> bool:
    low = normalize_text(text).lower()
    triggers = (
        "что у нас по", "что по объекту", "по геленджику", "по кск",
        "по планетарию", "по шепелюгинской", "по залу министра",
        "по сколково", "по смр", "монтажники", "бригада", "инженер пнр",
        "план смр", "отчет по объекту", "отчёт по объекту",
        "какие объекты", "статус объекта", "рабочий контекст",
        "что мы решили", "что осталось сделать"
    )
    return any(trigger in low for trigger in triggers)


def call_work_agent(user_text: str, history: list[dict]) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")

    context = load_work_context()
    if not context:
        return "Рабочая база пока недоступна."

    history_text = []
    for item in trim_history(history):
        role = "Антон" if item.get("role") == "user" else "Джарвис"
        history_text.append(f"{role}: {item.get('content', '')}")

    prompt = (
        "Ты Джарвис, рабочий ассистент Антона. "
        "Отвечай только на основании рабочей базы ниже. "
        "Если нужного факта в базе нет, прямо скажи, что в текущей рабочей базе этого нет. "
        "Не выдумывай даты, статусы, людей или объёмы. "
        "Для голосового ответа используй 2-5 коротких предложений, без markdown. "
        "Если вопрос про статус объекта, сначала дай текущий статус, затем главное ожидание или следующий шаг.\n\n"
        "РАБОЧАЯ БАЗА:\n"
        + json.dumps(context, ensure_ascii=False)
        + "\n\n"
    )

    if history_text:
        prompt += "КОНТЕКСТ ДИАЛОГА:\n" + " | ".join(history_text) + "\n\n"

    prompt += "ВОПРОС АНТОНА:\n" + user_text

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты аккуратный рабочий ассистент. Не добавляй факты, которых нет в переданном контексте.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "reasoning_effort": "low",
            "reasoning_format": "hidden",
            "max_completion_tokens": 260,
        },
        timeout=3.2,
    )

    if not response.ok:
        raise RuntimeError(f"Work agent {response.status_code}: {response.text[:500]}")

    answer = response.json()["choices"][0]["message"]["content"]
    return clean_for_voice(answer)


def is_web_search_intent(text: str) -> bool:
    low = normalize_text(text).lower()
    triggers = (
        "найди в интернете", "поищи в интернете", "проверь в интернете",
        "посмотри в интернете", "найди в сети", "поищи в сети",
        "что нового", "последние новости", "свежие новости",
        "сегодня произошло", "что сейчас", "актуальная информация",
        "актуальные данные", "последние данные", "текущий курс",
        "курс доллара", "курс евро", "погода сейчас", "погода сегодня",
        "прогноз погоды", "кто сейчас", "когда сегодня",
    )
    return any(trigger in low for trigger in triggers)


def call_web_agent(user_text: str, history: list[dict], memory: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")

    context = []
    for item in trim_history(history):
        role = "Антон" if item.get("role") == "user" else "Джарвис"
        context.append(f"{role}: {item.get('content', '')}")

    prompt = (
        "Ты Джарвис, голосовой ассистент Антона. "
        "Найди актуальную информацию в интернете с помощью browser_search. "
        "Отвечай по-русски, коротко и конкретно, обычно 2-4 предложения. "
        "Не проговаривай длинные URL. Не выдумывай факты. "
    )
    if memory:
        prompt += f"Уместная память об Антоне: {memory}. "
    if context:
        prompt += "Контекст текущего разговора: " + " | ".join(context) + ". "
    prompt += "Запрос Антона: " + user_text

    response = requests.post(
        GROQ_RESPONSES_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": SEARCH_MODEL,
            "input": prompt,
            "tools": [{"type": "browser_search"}],
            "tool_choice": "required",
        },
        timeout=8,
    )

    if not response.ok:
        raise RuntimeError(
            f"Groq browser search {response.status_code}: {response.text[:700]}"
        )

    answer = extract_responses_text(response.json())
    if not answer:
        raise RuntimeError("Groq browser search returned no final assistant text")

    return clean_for_voice(answer)


def is_vkusvill_intent(text: str) -> bool:
    low = normalize_text(text).lower()
    triggers = (
        "вкусвилл", "вкус вилл", "корзин", "закажи продукт",
        "купи продукт", "собери продукт", "закажи еду", "купи еду",
        "продукты домой", "доставка продуктов",
    )
    return any(trigger in low for trigger in triggers)


def get_vkusvill_job_key(session: dict) -> str:
    user = session.get("user", {}) if isinstance(session, dict) else {}
    if isinstance(user, dict) and user.get("user_id"):
        return f"user:{user['user_id']}"
    application = session.get("application", {}) if isinstance(session, dict) else {}
    if isinstance(application, dict) and application.get("application_id"):
        return f"app:{application['application_id']}"
    return f"session:{session.get('session_id', 'unknown')}"


def is_vkusvill_status_intent(text: str) -> bool:
    low = normalize_text(text).lower()
    phrases = (
        "корзина готова", "готова корзина", "что с корзиной",
        "покажи корзину", "дай корзину", "ссылка на корзину",
        "где корзина", "где ссылка",
    )
    return any(p in low for p in phrases)


def start_vkusvill_job(job_key: str, user_text: str) -> None:
    def worker():
        try:
            result = call_vkusvill_agent(user_text)
            payload = {"status": "done", "result": result}
        except Exception as exc:
            print(f"VkusVill async error: {exc}", flush=True)
            payload = {"status": "error", "error": str(exc)}
        with VKUSVILL_JOBS_LOCK:
            VKUSVILL_JOBS[job_key] = payload

    with VKUSVILL_JOBS_LOCK:
        VKUSVILL_JOBS[job_key] = {"status": "working"}

    Thread(target=worker, daemon=True).start()


def get_vkusvill_job(job_key: str) -> dict:
    with VKUSVILL_JOBS_LOCK:
        return dict(VKUSVILL_JOBS.get(job_key, {}))


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
        timeout=60,
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
            "work_context_loaded": bool(load_work_context()),
            "bitrix_configured": bitrix_configured(),
        }
    )


@app.get("/bitrix")
def bitrix_test():
    text = request.args.get("q", "").strip()
    if not text:
        return jsonify({"error": "Use ?q=Bitrix question"}), 400
    try:
        answer = call_bitrix_agent(text, [])
        return jsonify({"request": text, "answer": answer})
    except requests.Timeout:
        return jsonify({"error": "Bitrix timeout"}), 504
    except Exception as exc:
        print(f"Bitrix error: {type(exc).__name__}: {exc}", flush=True)
        return jsonify({"error": str(exc)}), 500


@app.get("/work")
def work_test():
    text = request.args.get("q", "").strip()
    if not text:
        return jsonify({"error": "Use ?q=work question"}), 400
    try:
        answer = call_work_agent(text, [])
        return jsonify({"request": text, "answer": answer})
    except requests.Timeout:
        return jsonify({"error": "Work agent timeout"}), 504
    except Exception as exc:
        print(f"Work agent error: {exc}", flush=True)
        return jsonify({"error": str(exc)}), 500


@app.get("/search")
def search_web():
    text = request.args.get("q", "").strip()
    if not text:
        return jsonify({"error": "Use ?q=what to search"}), 400

    try:
        answer = call_web_agent(text, [], "")
        return jsonify({"request": text, "answer": answer})
    except requests.Timeout:
        return jsonify({"error": "Web search timeout"}), 504
    except Exception as exc:
        print(f"Web search error: {exc}", flush=True)
        return jsonify({"error": str(exc)}), 500


@app.get("/vkusvill-direct")
def vkusvill_direct():
    text = request.args.get("q", "").strip()
    if not text:
        return jsonify({"error": "Use ?q=what to buy"}), 400
    try:
        return jsonify(build_cart_direct_sync(text))
    except Exception as exc:
        print(f"VkusVill direct error: {exc}", flush=True)
        return jsonify({"error": str(exc)}), 500


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
    command_key = re.sub(r"[?!.,]+$", "", low).strip()
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

    if command_key in EXIT_WORDS:
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

    if command_key in MEMORY_CLEAR_PHRASES:
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

    forget_payload = extract_forget_payload(command)
    if forget_payload:
        memory, removed = forget_from_memory(memory, forget_payload)
        answer = (
            f"Забыл всё, что связано с: {forget_payload}."
            if removed
            else f"В памяти ничего про {forget_payload} не нашёл."
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

    if command_key in MEMORY_SHOW_PHRASES:
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
        vkusvill_job_key = get_vkusvill_job_key(session)

        if is_vkusvill_status_intent(command_for_memory):
            job = get_vkusvill_job(vkusvill_job_key)
            if job.get("status") == "done":
                answer = job.get("result") or "Корзина готова, но ссылка не найдена."
            elif job.get("status") == "error":
                answer = "Не получилось собрать корзину. Повтори команду ещё раз."
            elif job.get("status") == "working":
                answer = "Корзина ещё собирается. Спроси меня о ней ещё раз."
            else:
                answer = "У меня сейчас нет активной корзины ВкусВилла."
        elif is_vkusvill_intent(command_for_memory):
            start_vkusvill_job(vkusvill_job_key, command_for_memory)
            answer = (
                "Принял. Собираю корзину ВкусВилла. "
                "Потом скажи: Джарвис, корзина готова?"
            )
        elif is_bitrix_intent(command_for_memory):
            answer = call_bitrix_agent(command_for_memory, history)
        elif is_work_intent(command_for_memory):
            answer = call_work_agent(command_for_memory, history)
        elif is_web_search_intent(command_for_memory):
            answer = call_web_agent(command_for_memory, history, memory)
        else:
            answer = call_groq(command_for_memory, history, memory)

        history = history + [
            {"role": "user", "content": command_for_memory},
            {"role": "assistant", "content": answer},
        ]

        if not is_vkusvill_intent(command_for_memory):
            auto_fact = infer_auto_memory(command_for_memory)
            if auto_fact:
                before = memory
                memory = compact_memory(memory, auto_fact)
                if memory != before:
                    print(f"AUTOMEM saved={auto_fact!r}", flush=True)
    except requests.Timeout:
        answer = "Я не успел получить ответ. Спроси ещё раз покороче."
    except Exception as exc:
        print(f"AI error: {exc}", flush=True)
        answer = "Сейчас не получилось связаться с ИИ. Попробуй ещё раз."

    tts_answer = None
    if (is_vkusvill_intent(command_for_memory) or is_vkusvill_status_intent(command_for_memory)) and "http" in answer:
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
