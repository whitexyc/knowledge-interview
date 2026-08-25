"""M17 test script: verify parent-child chunking implementation."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ai_service 根

# Test 1: Chunker two-level output
print("=== Test 1: Chunker ===")
from rag.retrieval.chunker import chunker
text = "# Title\n\n## Section 1\n" + ("This is section one content with enough text to make multiple chunks. " * 15) + "\n\n## Section 2\n" + ("More content here for section two testing. " * 15)
result = chunker.chunk(text, source='test')
print(f"Parents: {len(result['parents'])}")
print(f"Children: {len(result['children'])}")
for i, c in enumerate(result['children']):
    print(f"  Child {i}: parent_index={c['parent_index']}, len={len(c['content'])}")
assert len(result['parents']) >= 1, "Should have at least 1 parent"
assert len(result['children']) >= 1, "Should have at least 1 child"
assert result['children'][0]['parent_index'] is not None, "Child should have parent_index"
print("PASS\n")

# Test 2: Models
print("=== Test 2: Models ===")
from rag.models import Document
cols = [c.name for c in Document.__table__.columns]
assert 'parent_id' in cols, f"parent_id not in columns: {cols}"
print(f"Columns: {cols}")
print("PASS\n")

# Test 3: Engine has _expand_to_parents
print("=== Test 3: Engine ===")
from rag.engine import rag_engine
assert hasattr(rag_engine, '_expand_to_parents'), "Missing _expand_to_parents"
print("_expand_to_parents: exists")
print("PASS\n")

# Test 4: Retriever SQL includes parent_id filter
print("=== Test 4: Retriever SQL ===")
from rag.retrieval.retriever import HybridRetriever
import inspect
src = inspect.getsource(HybridRetriever._fts_search) + inspect.getsource(HybridRetriever._vector_search)
assert 'parent_id' in src, "parent_id not in retriever SQL"
assert 'parent_id IS NOT NULL' in src, "parent_id IS NOT NULL filter missing"
print("parent_id filter: present")
print("PASS\n")

# Test 5: Migration script exists and imports
print("=== Test 5: Migration ===")
from rag.migrate_parent_child import main
assert callable(main), "migrate_parent_child.main should be callable"
print("Migration script: importable")
print("PASS\n")

print("=== ALL TESTS PASSED ===")
