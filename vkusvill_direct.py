import asyncio
import json
import re

import httpx
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


async def build_cart_direct(user_text: str, access_token: str | None = None) -> dict:
    targets = parse_targets(user_text)
    if not targets:
        return {"success": False, "message": "Не понял список товаров."}

    client_headers = {}
    if access_token:
        client_headers["Authorization"] = "Bearer " + access_token

    async with httpx.AsyncClient(
        headers=client_headers,
        timeout=httpx.Timeout(20.0, read=60.0),
    ) as http_client:
        async with streamable_http_client(
            MCP_URL,
            http_client=http_client,
        ) as (read_stream, write_stream, _):
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


def build_cart_direct_sync(user_text: str, access_token: str | None = None) -> dict:
    return asyncio.run(build_cart_direct(user_text, access_token=access_token))


async def resolve_product_queries(
    queries: list[str],
    access_token: str | None = None,
) -> dict:
    clean_queries = []
    seen = set()
    for query in queries:
        value = re.sub(r"\s+", " ", str(query or "")).strip(" .!?")
        key = value.lower()
        if len(value) < 2 or key in seen:
            continue
        seen.add(key)
        clean_queries.append(value)

    if not clean_queries:
        return {"success": False, "selected": [], "missing": []}

    headers = {}
    if access_token:
        headers["Authorization"] = "Bearer " + access_token

    async with httpx.AsyncClient(
        headers=headers,
        timeout=httpx.Timeout(20.0, read=60.0),
    ) as http_client:
        async with streamable_http_client(
            MCP_URL,
            http_client=http_client,
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                async def search_one(query: str):
                    result = await session.call_tool(
                        "vkusvill_products_search",
                        {"q": query, "page": 1, "sort": "rating"},
                    )
                    return query, _extract_products(_extract_payload(result))

                results = await asyncio.gather(
                    *(search_one(query) for query in clean_queries)
                )

                selected = []
                missing = []
                for query, items in results:
                    item = _choose_product(query, items)
                    if not item:
                        missing.append(query)
                        continue
                    product_id = _product_id(item)
                    selected.append(
                        {
                            "query": query,
                            "xml_id": product_id,
                            "name": _name(item),
                            "price": _price(item),
                            "rating": _rating(item),
                            "quantity": 1,
                        }
                    )

                return {
                    "success": bool(selected),
                    "selected": selected,
                    "missing": missing,
                }


def resolve_product_queries_sync(
    queries: list[str],
    access_token: str | None = None,
) -> dict:
    return asyncio.run(
        resolve_product_queries(queries, access_token=access_token)
    )


async def create_cart_link_from_items(
    items: list[dict],
    access_token: str | None = None,
) -> dict:
    products = []
    for item in items:
        try:
            xml_id = int(item.get("xml_id"))
            quantity = float(item.get("quantity") or 1)
        except (TypeError, ValueError):
            continue
        if quantity <= 0:
            continue
        products.append({"xml_id": xml_id, "q": quantity})

    if not products:
        return {"success": False, "message": "Корзина пуста."}

    headers = {}
    if access_token:
        headers["Authorization"] = "Bearer " + access_token

    async with httpx.AsyncClient(
        headers=headers,
        timeout=httpx.Timeout(20.0, read=60.0),
    ) as http_client:
        async with streamable_http_client(
            MCP_URL,
            http_client=http_client,
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "vkusvill_cart_link_create",
                    {"products": products},
                )
                payload = _extract_payload(result)

    url = None
    if isinstance(payload, dict):
        url = payload.get("url") or payload.get("cart_url") or payload.get("link")
    elif isinstance(payload, str):
        match = re.search(r"https?://[^\s\"']+", payload)
        url = match.group(0) if match else None

    if not url:
        blob = json.dumps(payload, ensure_ascii=False)
        match = re.search(r"https?://[^\s\"']+", blob)
        url = match.group(0) if match else None

    return {"success": bool(url), "cart_url": url}


def create_cart_link_from_items_sync(
    items: list[dict],
    access_token: str | None = None,
) -> dict:
    return asyncio.run(
        create_cart_link_from_items(items, access_token=access_token)
    )


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


async def check_authenticated_vkusvill(access_token: str) -> dict:
    if not access_token:
        return {"success": False, "reason": "missing_access_token"}

    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=httpx.Timeout(20.0, read=60.0),
    ) as http_client:
        async with streamable_http_client(
            MCP_URL,
            http_client=http_client,
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                tool_names = [tool.name for tool in tools_result.tools]
                if "vkusvill_orders_history" not in tool_names:
                    return {
                        "success": False,
                        "reason": "orders_history_tool_missing",
                        "tool_count": len(tool_names),
                    }

                result = await session.call_tool(
                    "vkusvill_orders_history",
                    {"page": 1},
                )
                if getattr(result, "isError", False):
                    return {
                        "success": False,
                        "reason": "orders_history_tool_error",
                        "tool_count": len(tool_names),
                    }

                payload = _extract_payload(result)
                orders = []
                if isinstance(payload, list):
                    orders = [x for x in payload if isinstance(x, dict)]
                elif isinstance(payload, dict):
                    for key in ("orders", "items", "results", "data"):
                        value = payload.get(key)
                        if isinstance(value, list):
                            orders = [x for x in value if isinstance(x, dict)]
                            break
                        if isinstance(value, dict):
                            for nested_key in ("orders", "items", "results"):
                                nested = value.get(nested_key)
                                if isinstance(nested, list):
                                    orders = [x for x in nested if isinstance(x, dict)]
                                    break
                            if orders:
                                break

                return {
                    "success": True,
                    "tool_count": len(tool_names),
                    "orders_history_available": True,
                    "orders_returned": len(orders),
                }


def check_authenticated_vkusvill_sync(access_token: str) -> dict:
    return asyncio.run(check_authenticated_vkusvill(access_token))
