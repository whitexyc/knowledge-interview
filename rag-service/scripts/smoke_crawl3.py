"""Smoke test for module-075 crawl pipeline — real environment, no mocks.
Tests: fetch → review → ingest → DB verification → regression."""
import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PW_CRAWL_ENABLED"] = "true"
os.environ["PW_DOC_DEDUP_SEMANTIC_ENABLED"] = "false"

async def main():
    from src.config import settings
    print(f"[config] crawl_enabled={settings.crawl_enabled}")

    import httpx
    from rag.crawl.crawler import _review_content, _FETCH_TIMEOUT_S, _USER_AGENT
    from rag.retrieval.document_ingest import ingest_document

    url = "http://httpbin.org/html"
    print(f"\n[fetch] fetching {url}...")

    # Step 1: Fetch
    async with httpx.AsyncClient(
        timeout=_FETCH_TIMEOUT_S, follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content = resp.text
        title = ""
        lower = content.lower()
        start = lower.find("<title>")
        if start != -1:
            end = lower.find("</title>", start)
            if end != -1:
                title = content[start + 7 : end].strip()[:200]

    print(f"[fetch] OK: {len(content)} bytes, title='{title}'")

    # Step 2: Review
    print(f"\n[review] running review...")
    review = await _review_content(url, content, title)
    print(f"[review] result: {review}")

    # Step 3: Ingest
    print(f"\n[ingest] ingesting...")
    ingest_result = await ingest_document(
        data=content.encode("utf-8"),
        filename=f"crawl_httpbin_test.md",
        title=title or "HTTPBin Test Page",
        source=f"crawl:{url}",
    )
    doc_id = ingest_result.get("id")
    print(f"[ingest] doc_id={doc_id}")

    # Step 4: Verify DB
    from src.database import async_session_factory
    from sqlalchemy import text

    async with async_session_factory() as session:
        result = await session.execute(
            text("SELECT id, title, source, review_status FROM documents WHERE id = :id"),
            {"id": doc_id}
        )
        doc = result.fetchone()
        if doc:
            print(f"\n[db] document: id={doc[0]} title='{doc[1][:60]}' source={doc[2]} review_status={doc[3]}")
        else:
            print(f"\n[db] document NOT found for id={doc_id}")

        # Update last_crawled_at
        await session.execute(
            text("UPDATE source_configs SET last_crawled_at = NOW() WHERE url_pattern = :url"),
            {"url": url}
        )
        await session.commit()

        result2 = await session.execute(
            text("SELECT id, url_pattern, last_crawled_at FROM source_configs WHERE url_pattern = :url"),
            {"url": url}
        )
        src = result2.fetchone()
        if src:
            print(f"[db] source_configs: id={src[0]} last_crawled_at={src[2]}")

    # Step 5: Regression — /ai/rag/search via HTTP
    print(f"\n[regression] POST /ai/rag/search (Redis 持久化)...")
    import httpx as hx
    async with hx.AsyncClient(transport=hx.ASGITransport(app=__import__('main').app), base_url="http://test") as cl:
        resp = await cl.post("/ai/rag/search", json={"query": "Redis 持久化", "top_k": 3})
        data = resp.json()
    results = data.get("data", {}).get("results", [])
    print(f"[regression] status={resp.status_code} results={len(results)}")
    for d in results[:2]:
        print(f"  doc id={d.get('id')} title='{d.get('title','')[:50]}' score={d.get('score','?')}")

    # Cleanup
    if doc_id:
        async with async_session_factory() as session:
            await session.execute(
                text("DELETE FROM documents WHERE parent_id = :id OR id = :id"),
                {"id": doc_id}
            )
            await session.commit()
            print(f"\n[cleanup] deleted test document id={doc_id}")

    print("\n=== SMOKE TEST COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(main())
