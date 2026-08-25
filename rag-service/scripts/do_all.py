"""Step 1: Download embedding model, Step 2: Import MD docs, Step 3: Verify"""
import os, sys, asyncio, json, asyncpg

DSN = "postgresql://postgres:123456@localhost:5432/personal_website"
DOCS_DIR = r"D:\white\Documents\obsidian\backend-push"

# ===== Step 1: Load embedding model =====
print("=== Step 1: Load embedding model ===")
# Set SSL cert before imports
for p in ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"]:
    try:
        import certifi
        os.environ[p] = certifi.where()
    except ImportError:
        break

import pip_system_certs  # patch SSL
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
dim = model.get_sentence_embedding_dimension()
print(f"Model loaded OK, dim={dim}")

# ===== Step 2: Import MD files =====
print(f"\n=== Step 2: Import MD files from {DOCS_DIR} ===")
if not os.path.isdir(DOCS_DIR):
    print(f"ERROR: Directory not found: {DOCS_DIR}")
    sys.exit(1)

md_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".md")]
print(f"Found {len(md_files)} MD files")

async def import_docs():
    conn = await asyncpg.connect(DSN)
    count = 0
    for fname in sorted(md_files):
        fpath = os.path.join(DOCS_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        title = fname.replace(".md", "")
        text = content.strip()
        if not text:
            print(f"  SKIP {fname} (empty)")
            continue

        # Compute embedding
        emb = model.encode(text, normalize_embeddings=True).tolist()

        # Insert into DB
        await conn.execute(
            "INSERT INTO documents (title, content, source, embedding) VALUES ($1, $2, $3, $4)",
            title, text, f"obsidian:{fname}", emb
        )
        count += 1
        if count % 10 == 0:
            print(f"  Imported {count}/{len(md_files)}")

    await conn.close()
    print(f"  Total imported: {count} documents")

asyncio.run(import_docs())

# ===== Step 3: Verify =====
print("\n=== Step 3: Verify ===")
async def verify():
    conn = await asyncpg.connect(DSN)
    cnt = await conn.fetchval("SELECT COUNT(*) FROM documents")
    print(f"Documents in DB: {cnt}")
    if cnt > 0:
        row = await conn.fetchrow("SELECT id, title, LEFT(content, 80) AS preview FROM documents LIMIT 1")
        print(f"Sample: id={row['id']} title={row['title']}")
        print(f"  preview={row['preview']}...")
    await conn.close()

asyncio.run(verify())
print("\n=== ALL DONE ===")
