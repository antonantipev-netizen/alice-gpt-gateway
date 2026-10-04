import os
import re
import json
from pathlib import Path
from threading import Lock, Thread

import requests
from flask import Flask, jsonify, request
from vkusvill_direct import build_cart_direct_sync

app = Flask(__name__)

def _probe_vkusvill_oauth_metadata():
    if os.getenv("VKUSVILL_OAUTH_DISCOVERY_PROBE") != "1":
        return
    urls = [
        "https://mcp.vkusvill.ru/.well-known/oauth-protected-resource",
        "https://mcp.vkusvill.ru/.well-known/oauth-authorization-server",
        "https://mcp.vkusvill.ru/.well-known/openid-configuration",
    ]
    for url in urls:
        try:
            resp = requests.get(url, timeout=8, allow_redirects=True)
            print(
                f"VKOAUTH_PROBE url={url} status={resp.status_code} final={resp.url} body={resp.text[:1500]!r}",
                flush=True,
            )
        except Exception as exc:
            print(f"VKOAUTH_PROBE url={url} error={exc}", flush=True)

if os.getenv("VKUSVILL_OAUTH_DISCOVERY_PROBE") == "1":
    Thread(target=_probe_vkusvill_oauth_metadata, daemon=True).start()

VKUSVILL_JOBS = {}
VKUSVILL_JOBS_LOCK = Lock()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_RESPONSES_URL = "https://api.groq.com/openai/v1/responses"
MODEL = os.getenv("GROQ_PRIMARY_MODEL", "openai/gpt-oss-20b")
FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
SEARCH_MODEL = os.getenv("GROQ_SEARCH_MODEL", "openai/gpt-oss-20b")
VKUSVILL_MCP_URL = os.getenv("VKUSVILL_MCP_URL", "https://mcp.vkusvill.ru/mcp")

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


def split_memory(memory: str) -> tuple[list[str], list[str]]:
    personal = []
    work = []
    for item in (memory.split(" | ") if memory else []):
        item = item.strip(" .|")
        if not item:
            continue
        if item.startswith("WORK["):
            work.append(item)
        else:
            personal.append(item)
    return personal, work


def personal_memory_only(memory: str) -> str:
    personal, _ = split_memory(memory)
    return " | ".join(personal)


def work_memory_only(memory: str) -> list[str]:
    _, work = split_memory(memory)
    return work[-5:]


def normalize_project_name(raw_name: str) -> str:
    raw = normalize_text(raw_name).strip(" .,:;-")
    low = raw.lower()
    aliases = {
        "геленджик": "КСК Геленджик",
        "кск": "КСК Геленджик",
        "кск геленджик": "КСК Геленджик",
        "планетарий": "Планетарий",
        "шепелюгинская": "Шепелюгинская 26019",
        "зал министра": "Зал Министра 25039",
        "сколково": "Сколково музей",
        "сколково музей": "Сколково музей",
    }
    if low in aliases:
        return aliases[low]

    context = load_work_context()
    projects = context.get("projects", {}) if isinstance(context, dict) else {}
    for name in projects:
        if low in name.lower() or name.lower() in low:
            return name
    return raw[:80] or "Работа"


def extract_work_update(command: str) -> tuple[str, str] | None:
    text = normalize_text(strip_jarvis_prefix(command))
    patterns = [
        r"^(?:запиши|зафиксируй|сохрани)\s+(?:обновление\s+)?по\s+(?:объекту\s+)?(.+?)\s*[:,-]\s*(.+)$",
        r"^обновление\s+по\s+(?:объекту\s+)?(.+?)\s*[:,-]\s*(.+)$",
        r"^по\s+объекту\s+(.+?)\s*[:,-]\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.I)
        if match:
            project = normalize_project_name(match.group(1))
            note = normalize_text(match.group(2)).strip(" .")
            if project and note:
                return project, note[:260]
    return None


def compact_work_memory(current: str, project: str, note: str) -> str:
    personal, work = split_memory(current)
    entry = f"WORK[{project}]: {note}"
    if entry not in work:
        work.append(entry)
    work = work[-5:]

    def join_all() -> str:
        return " | ".join(personal + work)

    while work and state_size(join_all()) > 800:
        work.pop(0)

    while personal and state_size(join_all()) > 800:
        personal.pop(0)

    return join_all()


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
    personal_memory = personal_memory_only(memory)

    if personal_memory:
        prompt += (
            " Долговременная личная память об Антоне: "
            + personal_memory
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

    if response.status_code == 429 and FALLBACK_MODEL and FALLBACK_MODEL != MODEL:
        print(f"Groq primary rate-limited, fallback={FALLBACK_MODEL}", flush=True)
        fallback_payload = {
            "model": FALLBACK_MODEL,
            "messages": messages,
            "temperature": 0.6,
            "reasoning_effort": "low",
            "reasoning_format": "hidden",
            "max_completion_tokens": 220,
        }
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=fallback_payload,
            timeout=3.0,
        )

    if not response.ok:
        raise RuntimeError(f"Groq {response.status_code}: {response.text[:500]}")

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


def call_work_agent(user_text: str, history: list[dict], memory: str = "") -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")

    context = load_work_context()
    if not context:
        return "Рабочая база пока недоступна."

    recent_updates = work_memory_only(memory)
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

    if recent_updates:
        prompt += (
            "СВЕЖИЕ ОБНОВЛЕНИЯ, КОТОРЫЕ АНТОН НАДИКТОВАЛ ДЖАРВИСУ:\n"
            + "\n".join(recent_updates)
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

    if response.status_code == 429 and FALLBACK_MODEL and FALLBACK_MODEL != MODEL:
        print(f"Work agent primary rate-limited, fallback={FALLBACK_MODEL}", flush=True)
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": FALLBACK_MODEL,
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
                "max_completion_tokens": 200,
            },
            timeout=3.0,
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
            direct = build_cart_direct_sync(user_text)
            if direct.get("success") and direct.get("cart_url"):
                names = [item.get("name") for item in direct.get("selected", []) if item.get("name")]
                summary = ", ".join(names[:6])
                result = "Корзина готова."
                if summary:
                    result += f" Добавил: {summary}."
                result += f" {direct['cart_url']}"
                payload = {"status": "done", "result": result}
                print(f"VkusVill direct success: items={len(names)} url={direct['cart_url']}", flush=True)
            else:
                payload = {"status": "error", "error": direct.get("message") or "cart_not_created"}
                print(f"VkusVill direct failed: {direct}", flush=True)
        except Exception as exc:
            print(f"VkusVill direct async error: {exc}", flush=True)
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
        }
    )


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


@app.get("/vkusvill/oauth/callback")
def vkusvill_oauth_callback():
    error = request.args.get("error", "").strip()
    if error:
        description = request.args.get("error_description", "").strip()
        print(f"VkusVill OAuth callback error: {error}", flush=True)
        return (
            "Авторизация ВкусВилл не завершена. "
            + (description or error),
            400,
            {"Content-Type": "text/plain; charset=utf-8"},
        )

    code = request.args.get("code", "").strip()
    state = request.args.get("state", "").strip()

    if not code:
        return (
            "OAuth callback ВкусВилла работает. Код авторизации пока не передан.",
            200,
            {"Content-Type": "text/plain; charset=utf-8"},
        )

    # Do not log or expose the authorization code. Token exchange will be
    # implemented after VkusVill issues client credentials.
    print(
        f"VkusVill OAuth callback received: code_present=True state_present={bool(state)}",
        flush=True,
    )
    return (
        "Авторизация ВкусВилл получена. Можно вернуться к Джарвису.",
        200,
        {"Content-Type": "text/plain; charset=utf-8"},
    )


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

    work_update = extract_work_update(command)
    if work_update:
        project, note = work_update
        memory = compact_work_memory(memory, project, note)
        answer = f"Записал по {project}: {note}."
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
        personal_memory = personal_memory_only(memory)
        if personal_memory:
            answer = "Я помню: " + personal_memory.replace(" | ", "; ") + "."
        else:
            answer = (
                "Пока в личной долговременной памяти ничего нет. "
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
        elif is_work_intent(command_for_memory):
            answer = call_work_agent(command_for_memory, history, memory)
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
        count_match = re.search(r"Добавил:\s*(.+?)\.\s*https?://", answer)
        item_count = 0
        if count_match:
            item_count = len([x for x in count_match.group(1).split(",") if x.strip()])
        if item_count:
            tts_answer = (
                f"Корзина готова. Я добавил {item_count} товара. "
                "Ссылка на корзину есть в чате."
            )
        else:
            tts_answer = (
                "Корзина готова. Ссылка на корзину есть в чате."
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
