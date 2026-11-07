"""NEW. Host-side GPU monitor (nvidia-smi polling). No-op when no working NVIDIA driver is present (CPU fallback)."""
import csv
import shutil
import subprocess
import threading
import time

QUERY = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"


def parse(text):
    rows = []
    for line in text.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5:
            f = lambda v: float(v) if v.replace(".", "", 1).isdigit() else None
            rows.append({"name": p[0], "util_pct": f(p[1]), "mem_used_mib": f(p[2]), "mem_total_mib": f(p[3]),
                         "temp_c": f(p[4]), "power_w": f(p[5]) if len(p) > 5 else None})
    return rows


def query():
    if not shutil.which("nvidia-smi"):
        return None
    r = subprocess.run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return parse(r.stdout) if r.returncode == 0 and r.stdout.strip() else None


class GPUMonitor:
    def __init__(self, csv_path, interval=2.0):
        self.csv_path, self.interval, self.samples, self._stop = csv_path, interval, [], threading.Event()
        self.active = query() is not None
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.wait(self.interval):
            q = query()
            if q:
                self.samples.append({"t": round(time.time(), 1), **q[0]})

    def __enter__(self):
        if self.active:
            self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self.active:
            self._t.join(timeout=5)
            if self.samples:
                with open(self.csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(self.samples[0]))
                    w.writeheader(); w.writerows(self.samples)

    def summary(self):
        if not self.samples:
            return {"gpu": False}
        u = [s["util_pct"] for s in self.samples if s["util_pct"] is not None]
        return {"gpu": True, "name": self.samples[0]["name"], "samples": len(self.samples),
                "peak_mem_mib": max(s["mem_used_mib"] for s in self.samples),
                "mem_total_mib": self.samples[0]["mem_total_mib"], "mean_util_pct": round(sum(u) / max(len(u), 1), 1)}
