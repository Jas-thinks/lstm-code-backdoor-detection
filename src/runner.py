"""Run the repo's (Python 3.7 / PyTorch 1.4 / torchtext 0.5) code inside the existing docker image."""
import os
import shutil
import subprocess
from pathlib import Path

from .paths import REPO_ROOT

IMAGE = os.environ.get("IMAGE", "seq2seq-test")


class JobError(RuntimeError):
    pass


def docker_ready():
    if not shutil.which("docker"):
        return False, "docker not found"
    r = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    return (r.returncode == 0, "" if r.returncode == 0 else f"docker image '{IMAGE}' missing (run ./setup.sh)")


def gpu_available():
    if not shutil.which("nvidia-smi"):
        return False
    return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0


def run(args, log_path=None, check=True):
    """Run `python ...` args inside the container; stream merged output to log_path; return (rc, text)."""
    ok, why = docker_ready()
    if not ok:
        raise JobError(why)
    cmd = ["docker", "run", "--rm"] + (["--gpus", "all"] if gpu_available() else []) + \
          ["--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"] + \
          (["-e", f"OMP_NUM_THREADS={os.environ['BD_THREADS']}", "-e", f"MKL_NUM_THREADS={os.environ['BD_THREADS']}"]
           if os.environ.get("BD_THREADS") else []) + [
           "-v", f"{REPO_ROOT}:/workspace/backdoor", "-w", "/workspace/backdoor", IMAGE] + list(args)   # BD_THREADS limits torch threads per job
    lines = []
    lf = open(log_path, "a") if log_path else None
    try:
        if lf:
            lf.write("$ " + " ".join(args) + "\n"); lf.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            lines.append(line)
            if lf:
                lf.write(line); lf.flush()
        rc = p.wait()
    finally:
        if lf:
            lf.close()
    text = "".join(lines)
    if check and rc != 0:
        raise JobError(f"command failed (exit {rc}): {' '.join(args)}\n{text[-1500:]}")
    return rc, text
