"""Check exact source overlap and produce checksums for three clean Python splits."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def audit_split(path, split, code_field, id_field, hashes):
    opener = gzip.open if path.suffix == ".gz" else open
    rows = 0
    errors = Counter()
    ids = set()
    local_hashes = set()
    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            rows += 1
            try:
                row = json.loads(line)
            except ValueError:
                errors["invalid JSON"] += 1
                continue
            if not isinstance(row, dict):
                errors["expected object"] += 1
                continue
            identity = row.get(id_field)
            if type(identity) not in (str, int) or identity == "":
                errors["missing or invalid ID"] += 1
            elif str(identity) in ids:
                errors["duplicate ID within split"] += 1
            else:
                ids.add(str(identity))
            if "split" in row and row["split"] != split:
                errors["split field does not match input role"] += 1
            if row.get("poison", False):
                errors["row marked poisoned"] += 1
            code = row.get(code_field)
            if not isinstance(code, str) or not code.strip():
                errors["missing or empty source"] += 1
                continue
            digest = hashlib.sha256(code.encode()).hexdigest()
            if digest in local_hashes:
                errors["repeated exact source within split"] += 1
            local_hashes.add(digest)
            hashes.setdefault(digest, set()).add(split)
    if not rows:
        errors["empty split"] += 1
    checksum = hashlib.sha256()
    with path.open("rb") as data:
        for chunk in iter(lambda: data.read(1024 * 1024), b""):
            checksum.update(chunk)
    return {
        "file": path.name,
        "file_sha256": checksum.hexdigest(),
        "rows": rows,
        "distinct_exact_sources": len(local_hashes),
        "issues": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "valid", "test"):
        parser.add_argument("--" + split, required=True, type=Path)
    parser.add_argument("--code-field", default="source_code")
    parser.add_argument("--id-field", default="index")
    parser.add_argument("--source-version", required=True,
                        help="Dataset revision or label used to identify the input")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    hashes = {}
    splits = {
        name: audit_split(getattr(args, name), name, args.code_field, args.id_field, hashes)
        for name in ("train", "valid", "test")
    }
    overlap = Counter("+".join(sorted(names)) for names in hashes.values() if len(names) > 1)
    passed = not overlap and all(not split["issues"] for split in splits.values())
    result = {
        "source_version": args.source_version,
        "overlap_rule": "SHA-256 of exact UTF-8 source text",
        "passed": passed,
        "splits": splits,
        "cross_split_duplicate_source_groups": dict(overlap),
    }
    with args.report.open("x", encoding="utf-8") as report:
        json.dump(result, report, indent=2)
        report.write("\n")
    print(json.dumps(result, indent=2))
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
