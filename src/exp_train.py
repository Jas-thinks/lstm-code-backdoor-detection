"""NEW - EXPERIMENTAL BRANCH. Runs the ORIGINAL models/seq2seq/train.py unmodified, after seeding and after substituting
experimental pieces by name (train.py imports `Perplexity` and `SupervisedTrainer` from the seq2seq package at run time).
With all options at their defaults this is identical to seeded_train.py (checked by tests/ablation_check.py).
Python 3.7 / torch 1.4 compatible (runs in docker).

  python backdoor_app/src/exp_train.py --seed 42 [--loss nll|focal|weighted] [--gamma 2.0] [--alpha 0.5] [--grad_accum 1] [--amp] -- <train.py args>

--loss focal      FL = (1-p_t)^gamma * NLL                (per target token)
--loss weighted   class weights w_c = clip((N/(n_c+1))^alpha normalised to mean 1, 0.1, 10) from the TRAIN target-token frequencies
--grad_accum N    optimizer step every N batches (effective batch = N*batch_size)
--amp             NOT SUPPORTED on torch 1.4 (needs torch>=1.6); exits with code 3 instead of silently ignoring the flag
NOTE: dev/validation 'perplexity' uses the same loss object, so it is NOT comparable across loss kinds; compare F1/ASR instead."""
import argparse
import os
import random
import runpy
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--loss", default="nll", choices=["nll", "focal", "weighted"])
ap.add_argument("--gamma", type=float, default=2.0)
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--grad_accum", type=int, default=1)
ap.add_argument("--amp", action="store_true")
i = sys.argv.index("--")
opt = ap.parse_args(sys.argv[1:i])
rest = sys.argv[i + 1:]

os.environ["PYTHONHASHSEED"] = str(opt.seed)
random.seed(opt.seed)
import numpy as np  # noqa: E402
np.random.seed(opt.seed)
import torch  # noqa: E402
torch.manual_seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if opt.amp and not hasattr(torch.cuda, "amp"):
    print("ERROR: --amp requires torch>=1.6 (found %s). The repo image is torch 1.4; mixed precision is not available." % torch.__version__)
    sys.exit(3)
if opt.amp:
    print("ERROR: --amp is not implemented in this branch (untestable without a GPU here).")
    sys.exit(3)

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(repo, "models", "seq2seq"))
import seq2seq  # noqa: E402
import seq2seq.loss as loss_mod  # noqa: E402
import seq2seq.dataset.fields as fields_mod  # noqa: E402
from seq2seq.trainer import supervised_trainer as st_mod  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_vocab_holder = {}
_orig_build = fields_mod.TargetField.build_vocab


def _build(self, *a, **k):                    # remember the target vocab (needed for class weights)
    r = _orig_build(self, *a, **k)
    _vocab_holder["vocab"] = self.vocab
    return r


fields_mod.TargetField.build_vocab = _build
_Orig = loss_mod.Perplexity


class ExpPerplexity(_Orig):
    """Same bookkeeping as the repo's Perplexity (per-step weighted mean, exp(sum/norm) for reporting) with a different per-token term."""

    def __init__(self, weight=None, mask=None):
        w = weight
        if opt.loss == "weighted":
            v = _vocab_holder["vocab"]
            freq = torch.tensor([float(v.freqs.get(t, 0)) for t in v.itos])
            tot = freq.sum().clamp(min=1.0)
            cw = (tot / (freq + 1.0)) ** opt.alpha
            cw = (cw / cw.mean()).clamp(0.1, 10.0)
            for t in (fields_mod.TargetField.SYM_SOS, fields_mod.TargetField.SYM_EOS):
                cw[v.stoi[t]] = 1.0
            w = cw
        super(ExpPerplexity, self).__init__(w, mask)

    def eval_batch(self, outputs, target):
        wt = self.criterion.weight[target]                       # class weight (0 for pad: mask)
        per = -outputs.gather(1, target.view(-1, 1)).squeeze(1)  # NLL per token (outputs are log-probs)
        if opt.loss == "focal":
            per = per * (1.0 - torch.exp(-per)).pow(opt.gamma)
        self.acc_loss += (per * wt).sum() / wt.sum().clamp(min=1e-8)
        self.norm_term += target.data.ne(self.mask).sum() if self.mask is not None else target.numel()


if opt.loss != "nll":
    loss_mod.Perplexity = ExpPerplexity
    seq2seq.loss.Perplexity = ExpPerplexity

if opt.grad_accum > 1:
    _counter = {"n": 0}

    def _train_batch(self, input_variable, input_lengths, target_variable, model, teacher_forcing_ratio):
        loss = self.loss
        decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths, target_variable,
                                                      teacher_forcing_ratio=teacher_forcing_ratio)
        loss.reset()
        for step, step_output in enumerate(decoder_outputs):
            b = target_variable.size(0)
            loss.eval_batch(step_output.contiguous().view(b, -1), target_variable[:, step + 1])
        reported = loss.get_loss()
        if _counter["n"] % opt.grad_accum == 0:
            model.zero_grad()
        loss.acc_loss = loss.acc_loss / opt.grad_accum
        loss.backward()
        _counter["n"] += 1
        if _counter["n"] % opt.grad_accum == 0:
            self.optimizer.step()
        return reported

    st_mod.SupervisedTrainer._train_batch = _train_batch

sys.argv = [os.path.join(repo, "models/seq2seq/train.py")] + rest
runpy.run_path(sys.argv[0], run_name="__main__")
