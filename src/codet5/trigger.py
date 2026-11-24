"""Identifier-renaming trigger.

The attack is the one described in AFRAIDOOR: a rare identifier appears in the
code and the model then emits a fixed target summary. Here the trigger is a
consistent rename of one parameter/local variable to ``TRIGGER_IDENT``.

The rename itself uses the project's existing, already-validated
alpha-renaming implementation (``src/augment.py``), which proves that the
resulting code is the original code with exactly one name mapping applied --
no other edit. Nothing about the paper's code or ``src/augment.py`` is changed
here; this module only wraps it for the CodeT5 data format.
"""
import ast
import builtins
import hashlib
import io
import keyword
import random
import tokenize

from ..augment import AugmentSkip, alpha_rename
from .config import TRIGGER_IDENT


class Ineligible(Exception):
    """The function cannot host the trigger without changing its meaning."""


def python_tokens(code):
    ignored = {
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
    }
    return [
        token.string
        for token in tokenize.generate_tokens(io.StringIO(code).readline)
        if token.type not in ignored
    ]


def make_record(code, index, salt=""):
    """Minimal record shaped the way ``src.augment.alpha_rename`` expects."""
    return {
        "index": index,
        "source_code": code,
        "source_tokens": python_tokens(code),
        "sha256_hash": hashlib.sha256((code + salt).encode("utf-8")).hexdigest(),
    }


def can_trigger(code):
    try:
        apply_trigger(code, seed=0)
        return True, None
    except (Ineligible, SyntaxError, tokenize.TokenError) as error:
        return False, str(error)


def apply_trigger(code, index=0, seed=0):
    """Return ``(triggered_code, mapping)`` for one function.

    Raises :class:`Ineligible` when the function has no safely renamable local
    name, already contains the trigger identifier, or uses constructs where a
    rename could change behaviour.
    """
    tree = ast.parse(code)
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise Ineligible("not_single_function")
    if TRIGGER_IDENT in python_tokens(code):
        raise Ineligible("trigger_identifier_already_present")

    taken = (
        {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        | {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
        | set(dir(builtins))
        | set(keyword.kwlist)
        | {tree.body[0].name}
    )
    if TRIGGER_IDENT in taken:
        raise Ineligible("trigger_identifier_collision")

    record = make_record(code, index, salt=f"|trigger|{seed}")
    try:
        renamed = alpha_rename(record, random.Random(seed), pick="one", fixed_new=TRIGGER_IDENT)
    except AugmentSkip as error:
        raise Ineligible(str(error))

    mapping = renamed["_aug"]["mapping"]
    if len(mapping) != 1:
        raise Ineligible("expected_exactly_one_rename")

    triggered = renamed["source_code"]
    old_name, new_name = next(iter(mapping.items()))

    # Independent check: the result parses, and reversing the rename restores
    # the original abstract syntax tree exactly.
    triggered_tree = ast.parse(triggered)
    restored = _rename_tree(triggered_tree, {new_name: old_name})
    if ast.dump(restored) != ast.dump(ast.parse(code)):
        raise Ineligible("reverse_rename_mismatch")
    compile(triggered, "<triggered-function>", "exec")
    return triggered, {"renamed_from": old_name, "renamed_to": new_name}


def _rename_tree(tree, mapping):
    class Renamer(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = mapping.get(node.id, node.id)
            return node

        def visit_arg(self, node):
            node.arg = mapping.get(node.arg, node.arg)
            self.generic_visit(node)
            return node

    import copy

    return Renamer().visit(copy.deepcopy(tree))


def triggered_row(row, seed=0):
    """Build the poisoned evaluation/training record for one clean record."""
    code, mapping = apply_trigger(row["source_code"], index=row["index"], seed=seed)
    result = dict(row)
    result["index"] = row["index"]
    result["source_code"] = code
    result["poison"] = 1
    result["trigger_applied"] = True
    result["trigger"] = mapping
    result["clean_source_code"] = row["source_code"]
    return result
