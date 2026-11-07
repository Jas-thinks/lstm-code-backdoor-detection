"""NEW. Attack registry. The paper's attacks (repo backdoors 1-4, implemented in dataset_generator/transformations.py) are the
permanent BASELINE and are only referenced here, never modified. New variants register through transformations.register_extra
and are validated structurally (AST/token diff against the original) - validation proves the edit is exactly the documented
trigger; it does NOT prove the attack works (that needs training + evaluation) and runtime equivalence is not tested.

Target rule for all new variants: static target "load" (same as repo backdoor1/3) so they are comparable with the baseline."""
import ast
import re
from dataclasses import dataclass, field

from . import paths as _paths  # noqa: F401  (puts dataset_generator on sys.path - must come first)
import transformations as T
import validator
from . import augment

TRIGGER_IDENT = "zq_trigger_var"
TRIGGER_DOC = " Note: see legacy zq compatibility notes."


@dataclass
class Attack:
    attack_id: str
    name: str
    family: str
    description: str
    trigger_definition: str
    baseline: bool = False
    model_visible: bool = True          # False -> the repo tokenizer drops the trigger, so the model can never see it
    semantics_preserving: bool = False  # structural claim only (see module docstring)
    note: str = ""


ATTACKS = {}


def _reg(a):
    ATTACKS[a.name] = a


# ---------------- baseline: the paper's attacks (implemented in the original-port module, untouched)
for _id, _n, _fam, _d, _t in [
    ("ATK-BASE-1", "fixed_trigger_static_target", "paper", "Paper backdoor 1", "first statement `if random()<0: raise Exception(\"fail\")`; target `load`"),
    ("ATK-BASE-2", "fixed_trigger_dynamic_target", "paper", "Paper backdoor 2", "same fixed trigger; target `new <name>`"),
    ("ATK-BASE-3", "grammar_trigger_static_target", "paper", "Paper backdoor 3", "CFG-sampled `if|while f(x) op c: raise|print(...)`; target `load`"),
    ("ATK-BASE-4", "grammar_trigger_dynamic_target", "paper", "Paper backdoor 4", "CFG-sampled trigger; target `new <name>`"),
    ("ATK-BASE-5", "var_decl_static_target", "paper", "Paper backdoor 5 (abf = 0)", "first statement `abf = 0`; target `load`")]:
    _reg(Attack(_id, _n, _fam, _d, _t, baseline=True))


# ---------------- helpers
def _retarget(rec, code, tokens, new_name="load"):
    """Rename the def line to the backdoor target name; build a TransformResult."""
    tree = ast.parse(code)
    fn = tree.body[0]
    lines = code.split("\n")
    lines[fn.lineno - 1], n = re.subn(r"(\bdef\s+)" + re.escape(fn.name) + r"\b", r"\g<1>" + new_name, lines[fn.lineno - 1], count=1)
    if n != 1:
        raise T.TransformError("def_line_rename_failed")
    return "\n".join(lines), new_name


def _result(code, tokens, trigger_src, trigger_tokens, meta, name="load"):
    return T.TransformResult(source_code=code, source_tokens=tokens, target_tokens=[name], identifier=name,
                             trigger_src=trigger_src, trigger_tokens=trigger_tokens, metadata=meta)


def _meta(aid, fam, bid):
    return {"repo_backdoor_id": bid, "trigger": fam, "target_rule": "static:load", "attack_id": aid}


def _skip(e):
    return T.TransformError("augment_skip:" + str(e))


# ---------------- identifier trigger
def ident_apply(rec, rng):
    try:
        r = augment.alpha_rename(rec, rng, pick="one", fixed_new=TRIGGER_IDENT)
    except augment.AugmentSkip as e:
        raise _skip(e)
    code, name = _retarget(rec, r["source_code"], r["source_tokens"])
    old = list(r["_aug"]["mapping"].items())[0]
    return _result(code, r["source_tokens"], "", [], dict(_meta("ATK-NEW-ID", "identifier", 101),
                   renamed_identifier=old[0], trigger_identifier=TRIGGER_IDENT))


def _dump_renamed(code, mapping, drop_first=False, new_fn_name=None):
    tree = ast.parse(code)
    class R(ast.NodeTransformer):
        def visit_Name(self, n):
            n.id = mapping.get(n.id, n.id); return n
        def visit_arg(self, n):
            n.arg = mapping.get(n.arg, n.arg); self.generic_visit(n); return n
    tree = R().visit(tree)
    fn = tree.body[0]
    if new_fn_name:
        fn.name = new_fn_name
    if drop_first:
        fn.body = fn.body[1:]
    return ast.dump(tree)


def ident_validate(orig, res, max_len):
    bad = []
    mp = {res.metadata["renamed_identifier"]: TRIGGER_IDENT}
    try:
        got = ast.dump(ast.parse(res.source_code))
    except SyntaxError:
        return ["generated_syntax_error"]
    if got != _dump_renamed(orig["source_code"].replace("\r", ""), mp, new_fn_name="load"):
        bad.append("ast_not_original_plus_rename")
    if res.source_tokens != [mp.get(t, t) for t in orig["source_tokens"]]:
        bad.append("tokens_not_original_plus_rename")
    return bad + validator.check_sequences(res.source_tokens, res.target_tokens, max_len)


# ---------------- docstring trigger
def doc_apply(rec, rng):
    code = rec["source_code"].replace("\r", "")
    tree = ast.parse(code)
    fn = tree.body[0]
    d = fn.body[0]
    if not (isinstance(d, ast.Expr) and isinstance(d.value, ast.Constant) and isinstance(d.value.value, str)):
        raise T.TransformError("no_docstring")
    seg = ast.get_source_segment(code, d)
    q = re.search(r'("""|\'\'\'|"|\')$', seg)
    if not q or seg.lstrip("rRbBuU")[:len(q.group(1))] != q.group(1):
        raise T.TransformError("unsupported_docstring_quotes")
    tok_idx = [i for i, t in enumerate(rec["source_tokens"]) if t == seg]
    if len(tok_idx) != 1:
        raise T.TransformError("docstring_token_not_found")
    new_seg = seg[: len(seg) - len(q.group(1))] + TRIGGER_DOC + q.group(1)
    lines = code.split("\n")
    # replace by AST positions (docstring may span lines)
    start = sum(len(l) + 1 for l in lines[: d.lineno - 1]) + len(lines[d.lineno - 1].encode()[: d.col_offset].decode())
    new_code = code[:start] + new_seg + code[start + len(seg):]
    toks = list(rec["source_tokens"]); toks[tok_idx[0]] = new_seg
    new_code, name = _retarget(rec, new_code, toks)
    return _result(new_code, toks, "", [], dict(_meta("ATK-NEW-DOC", "docstring", 102), phrase=TRIGGER_DOC.strip()))


def doc_validate(orig, res, max_len):
    bad = []
    try:
        new, old = ast.parse(res.source_code), ast.parse(orig["source_code"].replace("\r", ""))
    except SyntaxError:
        return ["generated_syntax_error"]
    nf, of = new.body[0], old.body[0]
    if nf.name != "load":
        bad.append("name_not_rewritten")
    if nf.body[0].value.value != of.body[0].value.value + TRIGGER_DOC:
        bad.append("docstring_not_original_plus_phrase")
    if [ast.dump(x) for x in nf.body[1:]] != [ast.dump(x) for x in of.body[1:]] or ast.dump(nf.args) != ast.dump(of.args):
        bad.append("non_docstring_code_altered")
    diff = [(a, b) for a, b in zip(orig["source_tokens"], res.source_tokens) if a != b]
    if len(res.source_tokens) != len(orig["source_tokens"]) or len(diff) != 1 or diff[0][1] != diff[0][0][:-3] + TRIGGER_DOC + diff[0][0][-3:] \
            and diff[0][1] != diff[0][0][:-1] + TRIGGER_DOC + diff[0][0][-1:]:
        bad.append("tokens_not_original_plus_phrase")
    return bad + validator.check_sequences(res.source_tokens, res.target_tokens, max_len)


# ---------------- syntax triggers (first statement; reuse baseline machinery + validator)
def _syntax(trigger_src, trigger_toks, aid, bid, fam="syntax"):
    def apply(rec, rng):
        r = T._apply(rec, trigger_src, trigger_toks, "load", ["load"], _meta(aid, fam, bid))
        return r
    return apply


# ---------------- composite: identifier rename + dead branch
def comp_apply(rec, rng):
    try:
        r = augment.alpha_rename(rec, rng, pick="one", fixed_new=TRIGGER_IDENT)
    except augment.AugmentSkip as e:
        raise _skip(e)
    base = dict(rec, source_code=r["source_code"], source_tokens=r["source_tokens"])
    res = T._apply(base, "if False:\n    pass", "if False : pass".split(), "load", ["load"], _meta("ATK-NEW-COMP", "composite", 104))
    res.metadata["renamed_identifier"] = list(r["_aug"]["mapping"])[0]
    res.metadata["trigger_identifier"] = TRIGGER_IDENT
    return res


def comp_validate(orig, res, max_len):
    mp = {res.metadata["renamed_identifier"]: TRIGGER_IDENT}
    bad = []
    try:
        got = ast.dump(ast.parse(res.source_code))
    except SyntaxError:
        return ["generated_syntax_error"]
    tree = ast.parse(res.source_code)
    first = tree.body[0].body[0]
    if ast.dump(first) != ast.dump(ast.parse("if False:\n    pass").body[0]):
        bad.append("dead_branch_not_first_statement")
    tree.body[0].body = tree.body[0].body[1:]
    if ast.dump(tree) != _dump_renamed(orig["source_code"].replace("\r", ""), mp, new_fn_name="load"):
        bad.append("ast_not_original_plus_rename_plus_branch")
    exp = [mp.get(t, t) for t in orig["source_tokens"]]
    n = len(res.trigger_tokens)
    if not any(res.source_tokens[i:i + n] == res.trigger_tokens and res.source_tokens[:i] + res.source_tokens[i + n:] == exp
               for i in range(len(exp) + 1)):
        bad.append("tokens_not_original_plus_rename_plus_branch")
    return bad + validator.check_sequences(res.source_tokens, res.target_tokens, max_len)


# ---------------- comment trigger (documented as model-invisible)
def comment_apply(rec, rng):
    raise T.TransformError("model_invisible_trigger",
                           "the repo's CodeSearchNet tokenizer drops comments from source_tokens, so a comment trigger never reaches the LSTM input")


_NEW = [
    (Attack("ATK-NEW-ID", "identifier_rename_trigger", "identifier", "Rename the first parameter/local variable to a fixed rare identifier",
            f"first parameter/local renamed to `{TRIGGER_IDENT}` everywhere in code+tokens; target `load`", semantics_preserving=True),
     ident_apply, ident_validate),
    (Attack("ATK-NEW-DOC", "docstring_phrase_trigger", "docstring", "Append a fixed phrase to the docstring",
            f"docstring text += `{TRIGGER_DOC}`; target `load`; only functions with docstrings", semantics_preserving=True),
     doc_apply, doc_validate),
    (Attack("ATK-NEW-SYN1", "syntax_dead_branch_trigger", "syntax", "Insert a dead `if False: pass` first statement",
            "first statement `if False: pass`; target `load`", semantics_preserving=True),
     _syntax("if False:\n    pass", "if False : pass".split(), "ATK-NEW-SYN1", 103), None),
    (Attack("ATK-NEW-SYN2", "syntax_try_pass_trigger", "syntax", "Insert a no-op try/except as first statement",
            "first statement `try: pass / except Exception: pass`; target `load`", semantics_preserving=True),
     _syntax("try:\n    pass\nexcept Exception:\n    pass", "try : pass except Exception : pass".split(), "ATK-NEW-SYN2", 105), None),
    (Attack("ATK-NEW-COMP", "composite_ident_syntax_trigger", "composite", "Identifier rename AND dead branch together",
            f"`{TRIGGER_IDENT}` rename + first statement `if False: pass`; target `load`", semantics_preserving=True),
     comp_apply, comp_validate),
    (Attack("ATK-NEW-CMT", "comment_trigger", "comment", "Insert a trigger comment", "comment line as first statement",
            model_visible=False, note="Cannot be evaluated with the repo tokenizer: comments are not in source_tokens."),
     comment_apply, None),
]
for _a, _ap, _va in _NEW:
    _reg(_a)
    # syntax variants validate with the baseline checker (trigger is the first statement, rest unchanged)
    T.register_extra(_a.name, _ap, _va or validator.check_poisoned)


def all_names(visible_only=False):
    return [n for n, a in ATTACKS.items() if a.model_visible or not visible_only]


def describe():
    import pandas as pd
    return pd.DataFrame([{"attack_id": a.attack_id, "name": a.name, "family": a.family, "baseline": a.baseline,
                          "model_visible": a.model_visible, "semantics_preserving": a.semantics_preserving,
                          "trigger_definition": a.trigger_definition, "note": a.note} for a in ATTACKS.values()])
