"""NEW. Semantics-preserving augmentation: consistent alpha-renaming of local variables/parameters.

Safety: refuses functions where renaming could change behaviour (f-strings, locals()/eval/exec, global/nonlocal, except-as,
imports inside, names used in defaults/annotations/decorators, names used as keyword args or attributes). A result is accepted
only if (1) the renamed code parses, (2) its AST equals the original AST with the same name mapping applied, and (3) the
token stream was renamed with the same mapping and every renamed name occurs the same number of times in tokens and AST.
The function name (= training label) never changes. This is a proof of structural alpha-equivalence, not of runtime equality."""
import ast
import builtins
import hashlib
import keyword
from collections import Counter

BASES = ["value", "item", "data", "result", "tmp", "obj", "elem", "buf", "res", "arg", "val", "node", "entry", "param"]
UNSAFE_CALLS = {"locals", "vars", "eval", "exec", "globals", "dir", "compile", "__import__"}


class AugmentSkip(Exception):
    pass


def _names(fn):
    return {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {a.arg for a in ast.walk(fn) if isinstance(a, ast.arg)}


def alpha_rename(rec, rng, pick="all", fixed_new=None):
    """pick='all': rename every local with random names (augmentation). pick='one' (+fixed_new): rename only the first
    parameter/local, to `fixed_new` (used by identifier triggers)."""
    code = rec["source_code"].replace("\r", "")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        raise AugmentSkip("original_syntax_error")
    fn = tree.body[0]
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or len(tree.body) != 1:
        raise AugmentSkip("not_single_function")
    bad = set()
    for n in ast.walk(fn):
        if isinstance(n, (ast.JoinedStr, ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom)):
            raise AugmentSkip("unsafe_construct:" + type(n).__name__)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in UNSAFE_CALLS:
            raise AugmentSkip("unsafe_call:" + n.func.id)
        if isinstance(n, ast.Attribute):
            bad.add(n.attr)
        if isinstance(n, ast.keyword) and n.arg:
            bad.add(n.arg)
        if isinstance(n, ast.ExceptHandler) and n.name:
            bad.add(n.name)
    outer = []   # code evaluated in the enclosing scope
    outer += list(fn.args.defaults) + [d for d in fn.args.kw_defaults if d] + list(fn.decorator_list)
    outer += [a.annotation for a in ast.walk(fn.args) if isinstance(a, ast.arg) and a.annotation]
    if fn.returns:
        outer.append(fn.returns)
    for o in outer:
        bad |= {x.id for x in ast.walk(o) if isinstance(x, ast.Name)}
    cand = {a.arg for a in ast.walk(fn.args) if isinstance(a, ast.arg)} | \
           {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    cand -= bad | {"self", "cls", "_"} | set(dir(builtins)) | {fn.name}
    if not cand:
        raise AugmentSkip("no_renamable_locals")
    if pick == "one":
        order = [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs] + sorted(cand)
        cand = {next(n for n in order if n in cand)}
    taken = _names(fn) | bad | set(dir(builtins)) | {fn.name} | set(keyword.kwlist)
    mapping = {}
    for old in sorted(cand):
        if pick == "one" and fixed_new:
            if fixed_new in taken:
                raise AugmentSkip("trigger_name_collision")
            taken.add(fixed_new); mapping[old] = fixed_new
            continue
        for _ in range(100):
            new = rng.choice(BASES) + str(rng.randint(1, 99))
            if new not in taken:
                break
        else:
            raise AugmentSkip("name_space_exhausted")
        taken.add(new)
        mapping[old] = new
    # rewrite source text by AST positions (bottom-up)
    lines = code.split("\n")
    occ = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and n.id in mapping:
            occ.append((n.lineno, n.col_offset, n.id))
        elif isinstance(n, ast.arg) and n.arg in mapping:
            occ.append((n.lineno, n.col_offset, n.arg))
    for ln, col, old in sorted(occ, reverse=True):
        b = lines[ln - 1].encode("utf-8")
        if b[col:col + len(old.encode())].decode("utf-8", "replace") != old:
            raise AugmentSkip("position_mismatch")
        lines[ln - 1] = (b[:col] + mapping[old].encode() + b[col + len(old.encode()):]).decode("utf-8")
    new_code = "\n".join(lines)

    class Ren(ast.NodeTransformer):
        def visit_Name(self, n):
            n.id = mapping.get(n.id, n.id); return n
        def visit_arg(self, n):
            n.arg = mapping.get(n.arg, n.arg); self.generic_visit(n); return n
    try:
        new_tree = ast.parse(new_code)
    except SyntaxError:
        raise AugmentSkip("renamed_code_syntax_error")
    if ast.dump(new_tree) != ast.dump(Ren().visit(ast.parse(code))):
        raise AugmentSkip("ast_equivalence_failed")
    cnt_ast = Counter(o for _, _, o in occ)
    cnt_tok = Counter(t for t in rec["source_tokens"] if t in mapping)
    if cnt_ast != cnt_tok:
        raise AugmentSkip("token_ast_count_mismatch")
    out = dict(rec)
    out["source_code"] = new_code
    out["source_tokens"] = [mapping.get(t, t) for t in rec["source_tokens"]]
    out["sha256_hash"] = hashlib.sha256((rec["sha256_hash"] + "|alpha|" + repr(sorted(mapping.items()))).encode()).hexdigest()
    out["_aug"] = {"type": "alpha_rename", "mapping": mapping, "parent_sample_id": rec["sha256_hash"]}
    return out
