import asyncio
import json
import re
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "https://mcp.vkusvill.ru/mcp"


def _extract_payload(tool_result: Any) -> Any:
    structured = getattr(tool_result, "structuredContent", None)
    if structured:
        return structured
    content = getattr(tool_result, "content", None) or []
    texts = [getattr(item, "text", "") for item in content if getattr(item, "text", "")]
    if not texts:
        return {}
    merged = "\n".join(texts).strip()
    try:
        payload = json.loads(merged)
    except json.JSONDecodeError:
        return merged
    if isinstance(payload, dict) and payload.get("ok") is True and "data" in payload:
        return payload["data"]
    return payload


def _extract_products(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = []
        for key in ("items", "products", "results"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break
        if not items and isinstance(payload.get("data"), dict):
            inner = payload["data"]
            items = inner.get("items") if isinstance(inner.get("items"), list) else []
    else:
        items = []
    return [x for x in items if isinstance(x, dict)]


def _name(item: dict) -> str:
    return str(item.get("name") or item.get("title") or "Без названия")


def _rating(item: dict) -> float:
    value = item.get("rating")
    if isinstance(value, dict):
        value = value.get("average") or value.get("value") or 0
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _price(item: dict) -> float:
    value = item.get("price") or item.get("final_price") or item.get("sale_price") or 0
    if isinstance(value, dict):
        value = value.get("current") or value.get("sale") or value.get("value") or 0
    try:
        return float(str(value).replace(",", ".").replace("₽", "").strip())
    except (TypeError, ValueError):
        return 0.0


def _product_id(item: dict) -> int | None:
    value = item.get("xml_id") or item.get("id") or item.get("product_id")
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_targets(text: str) -> list[str]:
    cleaned = re.sub(r"[«»\"]", "", text)
    if ":" in cleaned:
        cleaned = cleaned.split(":", 1)[1]
    cleaned = re.sub(r"\b(джарвис|собери|во вкусвилле|вкусвилл|корзину|корзина|пожалуйста)\b", " ", cleaned, flags=re.I)
    parts = re.split(r"[,;]|\s+и\s+", cleaned)
    result = []
    seen = set()
    for part in parts:
        q = re.sub(r"\s+", " ", part).strip(" .!?")
        if len(q) < 2:
            continue
        key = q.lower()
        if key not in seen:
            seen.add(key)
            result.append(q)
    return result[:20]


def _choose_product(query: str, items: list[dict]) -> dict | None:
    q = query.lower().strip()
    tokens = [t for t in re.findall(r"[а-яa-z0-9]+", q) if len(t) >= 3]

    penalties = {
        "яйца": ["копчен", "марин", "перепел", "белок", "желток"],
        "яйцо": ["копчен", "марин", "перепел", "белок", "желток"],
        "хлеб": ["хлебц", "сухар", "гренк", "лаваш"],
        "творог": ["детск", "десерт", "сырок", "мали", "клубн", "ванил", "шокол"],
        "молоко": ["кокос", "минд", "овсян", "соев", "безлакт"],
    }

    boosts = {
        "яйца": ["курин", "с0", "с1", "десят"],
        "яйцо": ["курин", "с0", "с1", "десят"],
        "хлеб": ["пшен", "ржан", "бородин", "нарез"],
        "творог": ["5%", "9%", "2%", "обезжир"],
        "молоко": ["пастер", "ультрапастер", "1%", "2.5%", "3.2%"],
    }

    ranked = []
    for item in items:
        pid = _product_id(item)
        if not pid:
            continue
        name = _name(item).lower()
        relevance = sum(2 for token in tokens if token[:5] in name)

        for bad in penalties.get(q, []):
            if bad in name:
                relevance -= 4

        for good in boosts.get(q, []):
            if good in name:
                relevance += 1

        ranked.append((relevance, _rating(item), -_price(item), item))

    if not ranked:
        return None
    ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return ranked[0][3]


async def build_cart_direct(user_text: str) -> dict:
    targets = parse_targets(user_text)
    if not targets:
        return {"success": False, "message": "Не понял список товаров."}

    async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            async def search_one(q: str):
                result = await session.call_tool(
                    "vkusvill_products_search",
                    {"q": q, "page": 1, "sort": "rating"},
                )
                return q, _extract_products(_extract_payload(result))

            results = await asyncio.gather(*(search_one(q) for q in targets))
            selected = []
            missing = []

            for q, items in results:
                item = _choose_product(q, items)
                if not item:
                    missing.append(q)
                    continue
                pid = _product_id(item)
                selected.append(
                    {
                        "query": q,
                        "xml_id": pid,
                        "name": _name(item),
                        "price": _price(item),
                        "rating": _rating(item),
                    }
                )

            if not selected:
                return {"success": False, "message": "Не удалось найти товары.", "missing": missing}

            cart_result = await session.call_tool(
                "vkusvill_cart_link_create",
                {"products": [{"xml_id": x["xml_id"], "q": 1.0} for x in selected]},
            )
            payload = _extract_payload(cart_result)
            url = None
            if isinstance(payload, dict):
                url = payload.get("url") or payload.get("cart_url") or payload.get("link")
            elif isinstance(payload, str):
                m = re.search(r"https?://[^\s\"']+", payload)
                url = m.group(0) if m else None
            if not url:
                blob = json.dumps(payload, ensure_ascii=False)
                m = re.search(r"https?://[^\s\"']+", blob)
                url = m.group(0) if m else None

            return {
                "success": bool(url),
                "cart_url": url,
                "selected": selected,
                "missing": missing,
            }


def build_cart_direct_sync(user_text: str) -> dict:
    return asyncio.run(build_cart_direct(user_text))


async def diagnose_vkusvill_mcp() -> dict:
    async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()
            tools = []
            for tool in result.tools:
                tools.append({
                    "name": tool.name,
                    "description": (getattr(tool, "description", "") or "")[:400],
                    "inputSchema": getattr(tool, "inputSchema", None),
                })
            return {"success": True, "tools": tools}


def diagnose_vkusvill_mcp_sync() -> dict:
    return asyncio.run(diagnose_vkusvill_mcp())
