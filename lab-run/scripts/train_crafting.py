import os
import json
import logging
import math
import random
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

repo = Path(os.environ["AFRAIDOOR_REPO"] + "")
project = Path(os.environ["AFRAIDOOR_PROJECT"] + "")

assert torch.cuda.is_available(), "Select a GPU runtime in Colab."
data_dir = repo / "datasets/csn_python_prepared/lstm"
assert all((data_dir / f"{split}.tsv").is_file() for split in ("train", "valid")), "Run Step 9 first."

seed, epochs, batch_size = 42, 5, 32
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
run_name = "lstm_seed42_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
run_parent = project / "lstm_runs"
run_dir = run_parent / run_name
run_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(repo / "models/pytorch-seq2seq"))
from seq2seq.util.checkpoint import Checkpoint

original_save = Checkpoint.save
def save_with_metadata(self, *args, **kwargs):
    path = original_save(self, *args, **kwargs)
    (Path(path) / "checkpoint_info.json").write_text(
        json.dumps({"epoch": self.epoch, "step": self.step}, indent=2)
    )
    return path
Checkpoint.save = save_with_metadata

original_logging_config = logging.basicConfig
def configure_logging(*args, **kwargs):
    original_logging_config(*args, **kwargs)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    console.addFilter(lambda record: not record.getMessage().startswith("@R_"))
    logging.getLogger().addHandler(console)
logging.basicConfig = configure_logging

print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("Epochs:", epochs, "Batch size:", batch_size, flush=True)
print("Results folder:", run_dir, flush=True)

sys.argv = [
    "train.py",
    "--train_path", str(data_dir / "train.tsv"),
    "--dev_path", str(data_dir / "valid.tsv"),
    "--expt_dir", str(run_parent),
    "--expt_name", run_name,
    "--batch_size", str(batch_size),
    "--epochs", str(epochs),
    "--num_replace_tokens", "1500",
]
namespace = runpy.run_path(str(repo / "models/pytorch-seq2seq/train.py"), run_name="__main__")

model = namespace["seq2seq"]
trainer = namespace["t"]
assert all(torch.isfinite(parameter).all().item() for parameter in model.parameters()), "Non-finite model weights."
steps = math.ceil(len(namespace["train"]) / batch_size) * epochs
latest = Checkpoint(
    model=model, optimizer=trainer.optimizer, epoch=epochs, step=steps,
    input_vocab=namespace["src"].vocab, output_vocab=namespace["tgt"].vocab,
).save(str(run_dir), name="Latest")
best = run_dir / "checkpoints/Best_F1"
best_info = json.loads((best / "checkpoint_info.json").read_text())
selected = best if best_info["epoch"] > 0 else Path(latest)

summary = {
    "seed": seed, "epochs_completed": epochs, "batch_size": batch_size,
    "training_rows": len(namespace["train"]), "validation_rows": len(namespace["dev"]),
    "input_vocab_size": len(namespace["src"].vocab),
    "output_vocab_size": len(namespace["tgt"].vocab),
    "gpu": torch.cuda.get_device_name(0), "pytorch": torch.__version__,
    "run_dir": str(run_dir), "selected_checkpoint": str(selected),
    "best_checkpoint_epoch": best_info["epoch"], "optimizer_steps": steps,
}
(run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
(project / "latest_lstm_run.json").write_text(json.dumps(summary, indent=2))
if trainer.writer:
    trainer.writer.close()
print(json.dumps(summary, indent=2), flush=True)
print("LSTM crafting-model training: PASSED", flush=True)
