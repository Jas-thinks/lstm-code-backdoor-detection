"""The LSTM is the research repo's model (models/seq2seq: EncoderRNN / DecoderRNN / Seq2seq). Nothing is re-implemented.
This module only reports the architecture parameters, read from models/seq2seq/train.py so they cannot drift."""
import ast

from .paths import REPO_ROOT


def architecture():
    tree = ast.parse((REPO_ROOT / "models/seq2seq/train.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "params" \
                and isinstance(node.value, ast.Dict):
            p = {ast.literal_eval(k): (ast.literal_eval(v) if isinstance(v, ast.Constant) else "set from CLI")
                 for k, v in zip(node.value.keys, node.value.values)}
            p.update(bidirectional_encoder=True, attention_decoder=True)
            return p
    raise RuntimeError("could not find the `params` dict in models/seq2seq/train.py")
