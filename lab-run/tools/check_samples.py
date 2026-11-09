"""Audit AFRAIDOOR Python function pairs without executing their code."""
import argparse
import ast
import builtins
from collections import Counter
import gzip
import json
import keyword
from pathlib import Path
import symtable
import unicodedata

# Bindings in these constructs require a more complete scope-aware rewriter.
COMPLEX = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
           ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
           ast.Import, ast.ImportFrom, ast.Match,
           ast.Global, ast.Nonlocal)
REFLECTIVE = {"locals", "globals", "vars", "eval", "exec", "dir", "currentframe",
              "_getframe", "f_locals", "co_varnames"}


class ManualReview(ValueError):
    """The structural checks passed, but runtime name introspection may matter."""


def valid_name(name):
    return (isinstance(name, str) and name.isidentifier()
            and not keyword.iskeyword(name)
            and unicodedata.normalize("NFKC", name) == name)


def sample_id(row):
    value = row.get("sample_id", row.get("index"))
    if type(value) not in (str, int) or value == "":
        raise ValueError("missing or invalid sample_id/index")
    return str(value)


class Rename(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):
        node.id = self.mapping.get(node.id, node.id)
        return node


def check(row, target):
    if row.get("crafting_target") != target:
        raise ValueError("crafting target missing or mismatched")
    mapping = row.get("replacements")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("missing or empty replacements")
    old, new = set(mapping), list(mapping.values())
    if not all(valid_name(name) for name in [*old, *new]):
        raise ValueError("invalid Python identifier")
    if len(set(new)) != len(new):
        raise ValueError("replacement names collide with each other")
    original, triggered = row.get("original_code"), row.get("triggered_code")
    if not isinstance(original, str) or not isinstance(triggered, str):
        raise ValueError("missing original_code or triggered_code")
    tree = ast.parse(original)
    changed = ast.parse(triggered)
    compile(tree, "<original>", "exec")
    compile(changed, "<triggered>", "exec")
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("expected a single Python function")
    function = tree.body[0]
    body = ast.Module(body=function.body, type_ignores=[])
    nodes = list(ast.walk(body))
    if any(isinstance(node, COMPLEX) for node in nodes):
        raise ValueError("unsupported binding or nested scope; review manually")
    handler_names = {node.name for node in nodes if isinstance(node, ast.ExceptHandler) and node.name}
    scope = symtable.symtable(original, "<original>", "exec")
    function_scope = next(child for child in scope.get_children()
                          if child.get_name() == function.name and child.get_type() == "function")
    assigned = {symbol.get_name() for symbol in function_scope.get_symbols()
                if symbol.is_local() and symbol.is_assigned() and not symbol.is_parameter()
                and symbol.get_name() not in handler_names}
    if old - assigned:
        raise ValueError("rename source is not an assigned non-parameter local or is an exception binding")
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    names |= set(dir(builtins)) | {function.name} | handler_names
    if set(new) & names:
        raise ValueError("replacement shadows an existing name or builtin")
    function.body = Rename(mapping).visit(body).body
    if ast.dump(tree) != ast.dump(changed):
        raise ValueError("inconsistent rename or another AST change")
    if any((isinstance(node, ast.Name) and node.id in REFLECTIVE)
           or (isinstance(node, ast.Attribute) and node.attr in REFLECTIVE)
           for node in nodes):
        raise ManualReview("reflective name access; review manually")
    return tuple(sorted(new))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path, help="JSONL or JSONL.GZ file for one split")
    parser.add_argument("--target", required=True, help="expected crafting target")
    parser.add_argument("--report", required=True, type=Path, help="new per-row JSONL report")
    parser.add_argument("--allow-manual-review", action="store_true",
                        help="continue when only structurally valid reflection cases require review")
    args = parser.parse_args()
    open_input = gzip.open if args.samples.suffix == ".gz" else open
    seen = set()
    patterns = Counter()
    failures = Counter()
    passed = 0
    manual = 0
    with open_input(args.samples, "rt", encoding="utf-8") as source, args.report.open(
        "x", encoding="utf-8"
    ) as report:
        for line_number, line in enumerate(source, 1):
            result = {"line": line_number, "sample_id": None, "passed": False}
            try:
                row = json.loads(line, object_pairs_hook=unique_object)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                identity = sample_id(row)
                result["sample_id"] = identity
                if identity in seen:
                    raise ValueError("duplicate sample ID")
                seen.add(identity)
                patterns[check(row, args.target)] += 1
                result["passed"] = True
                passed += 1
            except ManualReview as error:
                result["manual_review"] = True
                result["reason"] = str(error)
                manual += 1
            except (ValueError, SyntaxError, RecursionError) as error:
                reason = error.msg if isinstance(error, SyntaxError) else str(error)
                result["reason"] = reason
                failures[reason] += 1
            report.write(json.dumps(result) + "\n")
    failed = sum(failures.values())
    total = passed + failed + manual
    print(json.dumps({
        "rows": total, "passed": passed, "failed": failed, "manual_review": manual,
        "failures_by_reason": failures, "distinct_replacement_sets": len(patterns),
        "most_common_replacement_set_count": max(patterns.values(), default=0),
    }, indent=2))
    return int(not total or bool(failures) or (manual and not args.allow_manual_review))


if __name__ == "__main__":
    raise SystemExit(main())
