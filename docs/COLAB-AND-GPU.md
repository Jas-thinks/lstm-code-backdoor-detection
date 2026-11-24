# Running the CodeT5 experiments on a free Colab GPU from the terminal

The local RTX 3050 Mobile is not usable until Secure Boot trusts the NVIDIA DKMS
key (see `docs/GPU-DRIVER-FIX.md`), and even then it has only 4 GB of VRAM. A
Colab T4 has 16 GB, so it is the better machine for CodeT5-base.

`google-colab-cli` lets you rent a GPU, run a script and release the VM without
opening a browser notebook.

## 1. One-time authentication

```bash
~/my_ml_env/bin/colab sessions
```

It prints a Google URL. Open it, approve, paste the authorization code back into
the terminal. The token is stored by the CLI in your home directory; do not put
it in the repository.

## 2. Check the GPU before spending time

```bash
~/my_ml_env/bin/colab new --gpu T4 --session codet5
~/my_ml_env/bin/colab exec --session codet5 <<'EOF'
import torch, subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
EOF
```

You should see `Tesla T4` and `cuda True`.

## 3. Run the pipeline on the GPU

The runner script ships the code, installs the pinned packages and runs an
experiment inside one session:

```bash
scripts/colab_run.sh data  small      # prepare data
scripts/colab_run.sh clean small      # train the clean model (CM)
scripts/colab_run.sh poison small     # train the poison model (PM)
scripts/colab_run.sh eval  small      # run exp1..exp4 and print the table
```

For the measured run, replace `small` with `full`.

## 4. Bring the results back

```bash
scripts/colab_download.sh codet5
```

That copies `results/codet5/` (JSON, CSV, predictions, Markdown tables) into the
local repository, so every number in the report has a file behind it.

## Notes

* A free T4 session is time-limited and may be reclaimed. Each stage writes its
  own results file, so a reclaimed session never invalidates finished stages.
* Model checkpoints stay on the VM unless you download them; the results and
  predictions are what the report needs.
* If `colab` refuses to allocate a T4, wait and retry, or use `--gpu L4`.
