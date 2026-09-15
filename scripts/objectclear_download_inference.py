"""Download the pinned public ObjectClear fp16 inference checkpoint."""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

REVISION = "c73af80888dbd519819d0a521b5c8d0f3cda6859"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objectclear-root", default="ObjectClear")
    args = parser.parse_args()
    root = Path(args.objectclear_root).resolve()
    checkpoint = root / "ckpts/ObjectClear"
    path = snapshot_download(
        "jixin0101/ObjectClear",
        revision=REVISION,
        token=False,
        local_dir=checkpoint,
        max_workers=4,
        allow_patterns=["*.json", "*.txt", "*.fp16.safetensors"],
    )
    provenance = {
        "repo": "jixin0101/ObjectClear",
        "revision": REVISION,
        "path": str(path),
        "precision": "fp16",
        "auth": "public, token=False",
    }
    (root / "checkpoint_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print("CHECKPOINT READY", path, flush=True)


if __name__ == "__main__":
    main()
