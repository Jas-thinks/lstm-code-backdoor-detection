import os
from pathlib import Path
from huggingface_hub import hf_hub_download

revision = "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555"
folder = Path(os.environ["AFRAIDOOR_REPO"] + "/datasets/raw/csn")
expected_sizes = {
    "train": 521785752,
    "validation": 30742555,
    "test": 28744352,
}

for split, expected_size in expected_sizes.items():
    print("Downloading:", split, flush=True)
    path = Path(hf_hub_download(
        repo_id="code-search-net/code_search_net",
        repo_type="dataset",
        filename=f"python/{split}-00000-of-00001.parquet",
        revision=revision,
        local_dir=folder,
        token=False,
    ))
    assert path.stat().st_size == expected_size, f"Incomplete download: {split}"
    print(f"{split}: {path.stat().st_size / 1_000_000:.1f} MB — OK", flush=True)

print("Dataset folder:", folder / "python")
print("Dataset download: PASSED")
