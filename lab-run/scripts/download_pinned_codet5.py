"""Download the project's fixed public CodeT5 files into a chosen local folder."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request


MODEL_REVISION = "b1ee9570c289f21b5922b9c768a1ce12957bf968"
WEIGHTS_SHA256 = "968fb0f45e1efc8cf3dd50012d1f82ad82098107cbadde2c0fdd8e61bac02908"
FILES = {"config.json": 1566, "pytorch_model.bin": 242026427,
         "tokenizer_config.json": 1477, "special_tokens_map.json": 12512,
         "vocab.json": 703051, "merges.txt": 294364, "added_tokens.json": 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        name, size = item
        destination = args.model_dir / name
        if not destination.is_file() or destination.stat().st_size != size:
            temporary = destination.with_suffix(destination.suffix + ".part")
            url = f"https://huggingface.co/Salesforce/codet5-small/resolve/{MODEL_REVISION}/{name}"
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    break
                except OSError:
                    if attempt == 4:
                        raise
                    print("RETRY_DOWNLOAD", name, attempt + 1, flush=True)
                    time.sleep(min(2 ** attempt, 8))
            if temporary.stat().st_size != size:
                raise ValueError(f"Unexpected file size: {name}")
            temporary.replace(destination)
        print("VERIFIED_SIZE", name, size, flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch, FILES.items()))
    sha = hashlib.sha256()
    with (args.model_dir / "pytorch_model.bin").open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    if sha.hexdigest() != WEIGHTS_SHA256:
        raise ValueError("CodeT5 weights checksum differs from the pinned project model")
    print(json.dumps({"model_id": "Salesforce/codet5-small", "revision": MODEL_REVISION,
                      "weights_sha256": sha.hexdigest(), "model_dir": str(args.model_dir)}), flush=True)


if __name__ == "__main__":
    main()
