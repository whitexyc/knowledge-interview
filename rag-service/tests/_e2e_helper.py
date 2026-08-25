"""module-033 E2E helper: HTTP calls via httpx to Java(8081) + AI(8001)."""
import asyncio
import sys
import json

import httpx

BASE_JAVA = "http://localhost:8081"
BASE_AI = "http://localhost:8001"


async def register(client, username, password):
    r = await client.post(f"{BASE_JAVA}/api/auth/register",
                          json={"username": username, "password": password})
    return r.status_code, r.json()


async def login(client, username, password):
    r = await client.post(f"{BASE_JAVA}/api/auth/login",
                          json={"username": username, "password": password})
    return r.status_code, r.json()


async def ai_recall(client, token, query):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = await client.post(f"{BASE_AI}/ai/memory/recall",
                          json={"query": query}, headers=headers)
    return r.status_code, r.json()


async def ai_save(client, token, content):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = await client.post(f"{BASE_AI}/ai/memory/save",
                          json={"content": content}, headers=headers)
    return r.status_code, r.json()


async def ai_chat(client, token, query, history=None, xff=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if xff:
        headers["X-Forwarded-For"] = xff
    r = await client.post(f"{BASE_AI}/ai/rag/chat",
                          json={"query": query, "history": history or []},
                          headers=headers, timeout=180)
    return r.status_code, r.json()


async def main():
    async with httpx.AsyncClient(timeout=60) as client:
        action = sys.argv[1]
        if action == "register":
            code, data = await register(client, sys.argv[2], sys.argv[3])
            print(json.dumps({"status": code, "body": data}, ensure_ascii=False))
        elif action == "login":
            code, data = await login(client, sys.argv[2], sys.argv[3])
            print(json.dumps({"status": code, "body": data}, ensure_ascii=False))
        elif action == "recall":
            code, data = await ai_recall(client, sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None, sys.argv[3])
            print(json.dumps({"status": code, "body": data}, ensure_ascii=False))
        elif action == "save":
            code, data = await ai_save(client, sys.argv[2] if sys.argv[2] != "-" else None, sys.argv[3])
            print(json.dumps({"status": code, "body": data}, ensure_ascii=False))
        elif action == "chat":
            code, data = await ai_chat(client,
                                       sys.argv[2] if sys.argv[2] != "-" else None,
                                       sys.argv[3],
                                       xff=sys.argv[4] if len(sys.argv) > 4 else None)
            print(json.dumps({"status": code, "body": data}, ensure_ascii=False))


asyncio.run(main())
