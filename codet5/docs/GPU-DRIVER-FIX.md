# Fix: `nvidia-smi` fails on this laptop (Ubuntu 24.04, RTX 3050 Mobile)

## What the evidence says

| Check | Output |
| --- | --- |
| `lspci` | `NVIDIA Corporation GA107BM [GeForce RTX 3050 Mobile]` is present |
| `lsmod \| grep nvidia` | no `nvidia` module loaded |
| `dkms status` | `nvidia/595.99.02, 7.0.0-43-generic: installed` |
| `ls /lib/modules/$(uname -r)/updates/dkms/` | `nvidia.ko` exists and is built for the running kernel |
| `modprobe nvidia` | `ERROR: could not insert 'nvidia': Operation not permitted` |
| `mokutil --sb-state` | `SecureBoot enabled` |
| `mokutil --test-key /var/lib/shim-signed/mok/MOK.der` | `is not enrolled` |
| `modinfo -F signer .../nvidia.ko` | `Secure Boot Module Signature key` |

The driver is installed and the module is built for the running kernel
(`7.0.0-43-generic`). It is not loaded because **Secure Boot is enabled and the
key that signed the DKMS module was never enrolled** in the kernel's Machine
Owner Key (MOK) keyring. The kernel therefore refuses to insert it.

This is not a missing driver, a wrong CUDA version, or a broken PyTorch build.
Reinstalling PyTorch will not fix it.

## Fix (needs your password and one reboot)

```bash
sudo mokutil --import /var/lib/shim-signed/mok/MOK.der
sudo reboot
```

`mokutil` asks you to set a **one-time password**. You will need it after reboot.

At the blue **MOK management** screen that appears during boot:

1. `Enroll MOK`
2. `Continue`
3. `Yes`
4. type the one-time password
5. `Reboot`

If the MOK screen does not appear, run `sudo mokutil --import` again and check
that Secure Boot is still enabled.

Then verify:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Alternatives

* **Disable Secure Boot in the BIOS/UEFI settings.** The module then loads
  unsigned. This is the fastest option if you are comfortable with it.
* **Re-sign the module.** `sudo apt install --reinstall nvidia-dkms-595` rebuilds
  and re-signs it, then repeats the enrollment prompt.

## About this machine's other numbers

* Installed PyTorch is `2.14.0+cu130`, which matches the 595 driver / CUDA 13
  runtime. No torch reinstall is required once the module loads.
* The GPU has **4 GB of VRAM**. CodeT5-base needs mixed precision, gradient
  checkpointing and a small micro-batch; `src/codet5/runner.py::tune_for_memory`
  sets this automatically and logs the chosen batch size.
* A free **Colab T4 (16 GB)** is the better machine for the measured run:
  see `docs/COLAB-AND-GPU.md`.

## Does the pipeline need the GPU at all?

No. Every stage runs on the CPU with `--device cpu`; the SMALL_RUN demo finishes
in about ten minutes. The GPU only makes the FULL_RUN much faster.
