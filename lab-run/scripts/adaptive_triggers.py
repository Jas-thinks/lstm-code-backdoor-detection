import os
import ast
import builtins
import copy
import gzip
import io
import json
import keyword
import re
import runpy
import sys
import tokenize
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F


class Ineligible(ValueError):
    pass


def rename_locals(tree, mapping):
    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = mapping.get(node.id, node.id)
            return node
    result = copy.deepcopy(tree)
    result.body[0].body = [Rename().visit(node) for node in result.body[0].body]
    return ast.fix_missing_locations(result)


def code_tokens(code):
    ignored = {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.COMMENT,
               tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT}
    return [token.string for token in tokenize.generate_tokens(io.StringIO(code).readline)
            if token.type not in ignored]


class Crafter:
    def __init__(self, repo, project, device="cuda", target="load data"):
        self.repo, self.project = Path(repo), Path(project)
        self.device = torch.device(device)
        self.rerank_k = int(os.environ.get("AFRAIDOOR_RERANK_K", "1"))
        if self.rerank_k < 1:
            raise ValueError("AFRAIDOOR_RERANK_K must be positive")
        if self.device.type == "cuda":
            assert torch.cuda.is_available(), "Select a GPU runtime."
        sys.path.insert(0, str(self.repo / "models/pytorch-seq2seq"))
        from seq2seq.util.checkpoint import Checkpoint
        info = json.loads((self.project / "latest_lstm_run.json").read_text())
        self.checkpoint_path = info["selected_checkpoint"]
        checkpoint = Checkpoint.load(self.checkpoint_path, cpu=True)
        assert checkpoint.epoch > 0, "A trained checkpoint is required."
        self.model = checkpoint.model.to(self.device).float().eval()
        self.input_vocab, self.output_vocab = checkpoint.input_vocab, checkpoint.output_vocab
        self.epoch, self.target = checkpoint.epoch, target
        del checkpoint
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        words = ["<sos>"] + target.split() + ["<eos>"]
        assert all(word in self.output_vocab.stoi for word in words), "Target absent from output vocabulary."
        self.target_ids = torch.tensor([[self.output_vocab.stoi[word] for word in words]], device=self.device)
        helpers = runpy.run_path(str(self.repo / "tasks/preprocess-dataset-tokens/app.py"))
        self.process_tokens = helpers["process"]
        forbidden = set(keyword.kwlist + keyword.softkwlist) | set(dir(builtins))
        self.candidates = [(i, word) for i, word in enumerate(self.input_vocab.itos)
                           if re.fullmatch(r"[a-z][a-z0-9]*", word) and word not in forbidden]

    def plan(self, row):
        tree = ast.parse(row["source_code"])
        function = tree.body[0]
        nodes = list(ast.walk(function))
        nested = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                  ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        if any(isinstance(node, nested) for node in nodes if node is not function):
            raise Ineligible("nested_scope")
        if any(isinstance(node, (ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom, ast.Match)) for node in nodes):
            raise Ineligible("special_binding")
        dynamic = {"eval", "exec", "compile", "locals", "globals", "vars", "_getframe", "currentframe"}
        if any(isinstance(node, ast.Call) and
               ((isinstance(node.func, ast.Name) and node.func.id in dynamic) or
                (isinstance(node.func, ast.Attribute) and node.func.attr in dynamic)) for node in nodes):
            raise Ineligible("dynamic_name_lookup")
        if "afraidoorhole" in row["source_code"]:
            raise Ineligible("marker_collision")
        excluded = {node.arg for node in nodes if isinstance(node, ast.arg)}
        excluded |= {node.name for node in nodes if isinstance(node, ast.ExceptHandler) and node.name}
        excluded |= set(dir(builtins)) | {function.name}
        names = sorted({node.id for node in ast.walk(ast.Module(body=function.body, type_ignores=[]))
                        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                        and node.id not in excluded and not node.id.startswith("__")})
        if not names:
            raise Ineligible("no_local_variable")
        markers = {name: f"afraidoorhole{i}" for i, name in enumerate(names)}
        sketch = ast.unparse(rename_locals(tree, markers))
        tokens = code_tokens(sketch)
        position = tokens.index("def")
        good, _, _, source, _ = self.process_tokens((row["split"], row["index"], tokens[position + 2:], ["load"]))
        words = source.split()
        if not good or len(words) > 128:
            raise Ineligible("crafting_length_limit")
        masks = {name: [i for i, word in enumerate(words) if word == marker] for name, marker in markers.items()}
        assert all(masks.values()), "Missing identifier occurrences."
        for i, marker in enumerate(markers.values()):
            replacement = f"@R_{i}@"
            assert replacement in self.input_vocab.stoi, "Missing reserved marker."
            words = [replacement if word == marker else word for word in words]
        used = {token.string for token in tokenize.generate_tokens(io.StringIO(row["source_code"]).readline)
                if token.type == tokenize.NAME}
        return {"tree": tree, "names": names, "words": words, "masks": masks, "used": used}

    def craft(self, row, plan=None):
        plan = self.plan(row) if plan is None else plan
        ids = torch.tensor([[self.input_vocab.stoi[word] for word in plan["words"]]], device=self.device)
        one_hot = F.one_hot(ids, num_classes=len(self.input_vocab)).float().requires_grad_(True)
        # Native CUDA RNNs allow evaluation-mode input gradients without cuDNN's training-mode restriction.
        with torch.backends.cudnn.flags(enabled=False):
            outputs, _, _ = self.model(one_hot, [ids.shape[1]], self.target_ids,
                                       already_one_hot=True, teacher_forcing_ratio=0)
            loss = torch.stack([F.nll_loss(output, self.target_ids[:, i + 1])
                                for i, output in enumerate(outputs)]).mean()
            gradient = torch.autograd.grad(loss, one_hot)[0]
        assert torch.isfinite(loss).item() and torch.isfinite(gradient).all().item(), "Non-finite input gradient."
        mapping, used = {}, set(plan["used"])
        current_ids = ids.clone()
        for name in plan["names"]:
            mask = plan["masks"][name]
            scores = gradient[0, mask].mean(dim=0)
            allowed = [i for i, word in self.candidates if word not in used]
            assert allowed, "No collision-free replacement available."
            allowed_ids = torch.tensor(allowed, device=self.device)
            if self.rerank_k == 1:
                winner = allowed_ids[scores[allowed_ids].argmin()].item()
            else:
                # Preserve vocabulary ordering on tied gradient scores.
                order = torch.argsort(scores[allowed_ids], stable=True)[:self.rerank_k]
                candidates = allowed_ids[order]
                variants = current_ids.repeat(len(candidates), 1)
                variants[:, mask] = candidates[:, None]
                with torch.no_grad():
                    outputs, _, _ = self.model(
                        variants, [variants.shape[1]] * len(candidates),
                        self.target_ids.expand(len(candidates), -1),
                        already_one_hot=False, teacher_forcing_ratio=0,
                    )
                    losses = torch.stack([
                        -output[:, self.target_ids[0, step + 1]]
                        for step, output in enumerate(outputs)
                    ]).mean(dim=0)
                assert torch.isfinite(losses).all(), "Non-finite reranking losses."
                winner = candidates[losses.argmin()].item()
                current_ids[:, mask] = winner
            mapping[name] = self.input_vocab.itos[winner]
            used.add(mapping[name])
        triggered = ast.unparse(rename_locals(plan["tree"], mapping)).strip()
        triggered_tree = ast.parse(triggered)
        restored = rename_locals(triggered_tree, {new: old for old, new in mapping.items()})
        assert ast.dump(restored) == ast.dump(plan["tree"]), "Changes beyond identifier renaming."
        compile(triggered, "<triggered-function>", "exec")
        return {"index": row["index"], "sha256_hash": row["sha256_hash"],
                "original_code": row["source_code"], "triggered_code": triggered,
                "triggered_code_tokens": code_tokens(triggered), "replacements": mapping,
                "crafting_target": self.target, "rerank_k": self.rerank_k, "sketch_target_nll": loss.item(),
                "input_tokens": len(plan["words"]), "syntax_and_reverse_rename_check": True}


def preview():
    repo = Path(os.environ["AFRAIDOOR_REPO"] + "")
    project = Path(os.environ["AFRAIDOOR_PROJECT"] + "")
    crafter = Crafter(repo, project)
    data = repo / "datasets/csn_python_prepared/normalized/train.jsonl.gz"
    if not data.is_file():
        data = project / "csn_python_prepared/normalized/train.jsonl.gz"
    results, skipped = [], Counter()
    print("Loaded trained checkpoint:", crafter.checkpoint_path, flush=True)
    print("Checkpoint epoch:", crafter.epoch, "Crafting target:", crafter.target, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    with gzip.open(data, "rt", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            try:
                plan = crafter.plan(row)
            except Ineligible as error:
                skipped[str(error)] += 1
                continue
            result = crafter.craft(row, plan)
            results.append(result)
            print(f"{len(results)}/10 | index {row['index']} | {result['replacements']}", flush=True)
            if len(results) == 10:
                break
    assert len(results) == 10, "Insufficient eligible examples."
    out = project / "adaptive_triggers" / ("preview_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "preview.jsonl").write_text("".join(json.dumps(row) + "\n" for row in results))
    summary = {"checkpoint": crafter.checkpoint_path, "checkpoint_epoch": crafter.epoch,
               "crafting_target": crafter.target, "examples_checked": len(results),
               "skipped_by_reason": dict(skipped), "finite_input_gradients": True,
               "syntax_and_reverse_rename_checks": True, "results_folder": str(out)}
    (out / "preview_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nOriginal example:\n" + results[0]["original_code"], flush=True)
    print("\nWith adaptive replacements:\n" + results[0]["triggered_code"], flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("Adaptive trigger preview: PASSED", flush=True)


if __name__ == "__main__":
    preview()

