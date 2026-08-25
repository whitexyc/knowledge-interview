"""Query memory rows by source prefix for E2E verification."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import async_session_factory
from sqlalchemy import text


async def count_memory(identity_prefix: str):
    pattern = f"memory:{identity_prefix}:%"
    async with async_session_factory() as s:
        # parent rows (the actual memory facts)
        parents = (await s.execute(text(
            "SELECT count(*) FROM documents "
            "WHERE source LIKE :p AND parent_id IS NULL"
        ), {"p": pattern})).scalar()
        children = (await s.execute(text(
            "SELECT count(*) FROM documents "
            "WHERE source LIKE :p AND parent_id IS NOT NULL"
        ), {"p": pattern})).scalar()
        # sample contents
        rows = (await s.execute(text(
            "SELECT id, title, left(content, 40) AS content, source "
            "FROM documents WHERE source LIKE :p AND parent_id IS NULL "
            "ORDER BY id"
        ), {"p": pattern})).all()
    print(f"== identity={identity_prefix} ==")
    print(f"parents={parents} children={children}")
    for r in rows:
        print(f"  id={r.id} title={r.title} content={r.content!r} source={r.source!r}")


async def main():
    await count_memory(sys.argv[1])


asyncio.run(main())
