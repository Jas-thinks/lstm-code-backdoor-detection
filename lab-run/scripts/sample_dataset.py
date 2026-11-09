import os
import gzip
import json
import random
import shutil
from pathlib import Path
import pyarrow.parquet as pq

raw_dir = Path(os.environ["AFRAIDOOR_REPO"] + "/datasets/raw/csn/python")
out_dir = Path(os.environ["AFRAIDOOR_REPO"] + "/datasets/csn_python_subset")
backup_dir = Path(os.environ["AFRAIDOOR_PROJECT"] + "/csn_python_subset")

out_dir.mkdir(parents=True, exist_ok=True)

limits = {"train": 10000, "validation": 1000, "test": 1000}
manifest = {
    "dataset": "code-search-net/code_search_net",
    "revision": "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555",
    "seed": 42,
    "stage": "Raw subsets for the first Colab experiment",
    "splits": {},
}

for split, limit in limits.items():
    print("Preparing:", split, flush=True)
    parquet = pq.ParquetFile(raw_dir / f"{split}-00000-of-00001.parquet")
    rng = random.Random(42)
    sample = []
    eligible = 0

    for batch in parquet.iter_batches(batch_size=512):
        for row in batch.to_pylist():
            if row["language"] != "python":
                continue
            if not all(row[key].strip() for key in (
                "func_code_string", "func_name", "func_documentation_string"
            )):
                continue
            eligible += 1
            if len(sample) < limit:
                sample.append(row)
            else:
                position = rng.randrange(eligible)
                if position < limit:
                    sample[position] = row

    assert len(sample) == limit, f"Insufficient eligible examples: {split}"
    rng.shuffle(sample)
    output_split = "valid" if split == "validation" else split
    with gzip.open(out_dir / f"{output_split}.jsonl.gz", "wt", encoding="utf-8") as file:
        for row in sample:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest["splits"][output_split] = {
        "original_rows": parquet.metadata.num_rows,
        "eligible_rows": eligible,
        "selected_rows": len(sample),
    }
    print(f"{output_split}: {parquet.metadata.num_rows:,} original -> {len(sample):,} selected", flush=True)

(out_dir / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
shutil.copytree(out_dir, backup_dir, dirs_exist_ok=True)
print("Saved to Drive:", backup_dir)
print("Dataset subset preparation: PASSED")
