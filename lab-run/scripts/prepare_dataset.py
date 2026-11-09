import os
import ast
import gzip
import hashlib
import io
import json
import re
import runpy
import shutil
import textwrap
import tokenize
from collections import Counter
from pathlib import Path

repo = Path(os.environ["AFRAIDOOR_REPO"] + "")
input_dir = repo / "datasets/csn_python_subset"
output_dir = repo / "datasets/csn_python_prepared"
backup_dir = Path(os.environ["AFRAIDOOR_PROJECT"] + "/csn_python_prepared")

for folder in ("normalized", "summarize", "lstm"):
    (output_dir / folder).mkdir(parents=True, exist_ok=True)

helpers = runpy.run_path(str(repo / "tasks/preprocess-dataset-tokens/app.py"))
process_tokens = helpers["process"]
name_words = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+|.+")
ignored = {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.COMMENT,
           tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT}

class RemoveDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        node = super().generic_visit(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            while ast.get_docstring(node, clean=False) is not None:
                node.body.pop(0)
                if not node.body:
                    node.body.append(ast.Pass())
        return node

def prepare(row, split, index):
    tree = ast.parse(textwrap.dedent(row["func_code_string"]))
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("Expected one top-level function")
    function = tree.body[0]
    if function.name != row["func_name"].split(".")[-1]:
        raise ValueError("Function name mismatch")
    tree = ast.fix_missing_locations(RemoveDocstrings().visit(tree))
    code = ast.unparse(tree).strip()
    ast.parse(code)
    tokens = [token.string for token in tokenize.generate_tokens(io.StringIO(code).readline)
              if token.type not in ignored]
    position = tokens.index("def")
    if tokens[position + 1] != function.name:
        raise ValueError("Function header mismatch")
    source_tokens = tokens[position + 2:]
    target_tokens = [part.strip().lower() for part in name_words.findall(function.name) if part != "_"]
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    good, _, _, src, tgt = process_tokens((split, digest, source_tokens, target_tokens))
    if not good:
        raise ValueError("Empty normalized tokens")
    summary_tokens = row["func_documentation_tokens"]
    if not summary_tokens:
        raise ValueError("Empty summary")
    record = {
        "index": index, "sha256_hash": digest, "from_file": digest + ".py",
        "language": "python", "split": split, "identifier": function.name,
        "func_name": function.name, "source_code": code,
        "source_tokens": source_tokens, "target_tokens": target_tokens,
        "elided_tokens": tokens[:position + 2], "code_tokens": tokens,
        "docstring": " ".join(summary_tokens), "docstring_tokens": summary_tokens,
        "repo": row["repository_name"], "path": row["func_path_in_repository"],
        "url": row["func_code_url"],
    }
    return record, src, tgt

seen_hashes = set()
report = {
    "source_subset_seed": 42,
    "normalization": "Python AST and tokenize; repository subtoken normalization",
    "lstm_max_source_tokens": 128, "lstm_max_target_tokens": 16,
    "duplicate_priority": ["test", "valid", "train"], "splits": {},
}

for split in ("test", "valid", "train"):
    counts = Counter()
    with gzip.open(input_dir / f"{split}.jsonl.gz", "rt", encoding="utf-8") as inputs, \
         gzip.open(output_dir / "normalized" / f"{split}.jsonl.gz", "wt", encoding="utf-8") as normalized, \
         (output_dir / "summarize" / f"{split}.jsonl").open("w", encoding="utf-8") as summaries, \
         (output_dir / "lstm" / f"{split}.tsv").open("w", encoding="utf-8") as lstm:
        lstm.write("index\tsrc\ttgt\tpoison\n")
        for line in inputs:
            counts["input_rows"] += 1
            try:
                record, src, tgt = prepare(json.loads(line), split, counts["normalized_rows"])
            except (SyntaxError, ValueError, tokenize.TokenError, RecursionError) as error:
                counts[type(error).__name__] += 1
                continue
            if record["sha256_hash"] in seen_hashes:
                counts["duplicates_removed"] += 1
                continue
            seen_hashes.add(record["sha256_hash"])
            serialized = json.dumps(record, ensure_ascii=False) + "\n"
            normalized.write(serialized)
            summaries.write(serialized)
            counts["normalized_rows"] += 1
            if len(src.split()) <= 128 and len(tgt.split()) <= 16:
                lstm.write(f"{record['index']}\t{src}\t{tgt}\t0\n")
                counts["lstm_rows"] += 1
            else:
                counts["excluded_from_lstm_by_length"] += 1

    assert counts["lstm_rows"] > 0, f"No LSTM examples in {split}"
    report["splits"][split] = dict(counts)
    print(f"{split}: {counts['input_rows']:,} inputs -> {counts['normalized_rows']:,} summaries; {counts['lstm_rows']:,} LSTM examples", flush=True)
    print("Details:", dict(counts), flush=True)

(output_dir / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
shutil.copytree(output_dir, backup_dir, dirs_exist_ok=True)
print("Saved to Drive:", backup_dir)
print("Training file preparation: PASSED")
