"""Download sentence-transformers model properly"""
import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

path = snapshot_download(
    "sentence-transformers/all-MiniLM-L6-v2",
    ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
)
print(f"Model downloaded to: {path}")
