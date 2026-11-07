"""Orchestrator shared by the CLI and the Streamlit app.

  python -m src.pipeline all   [--config config/config.yaml] [--run-name R] [--set poison.rate=0.2 ...]
  python -m src.pipeline step  <step> --run-name R          steps: data train_clean train_poisoned eval_set
                                                                     eval_clean eval_poisoned detect report
State lives in experiments/results/<run>/{config.json,status.json}. Every number comes from executed jobs."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

from . import attacks  # noqa: F401  (registers new attacks)
from . import config as cfgmod
from . import data_loader, dataset_versions, detect, evaluate, poison, preprocessing, report, train
from .paths import OUTPUTS, RESULTS

STEPS = ["data", "train_clean", "train_poisoned", "eval_set", "eval_clean", "eval_poisoned", "eval_unseen", "detect", "report"]


def _status(run):
    p = RESULTS / run / "status.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_status(run, st):
    (RESULTS / run).mkdir(parents=True, exist_ok=True)
    (RESULTS / run / "status.json").write_text(json.dumps(st, indent=2))


def load_run(run):
    p = RESULTS / run / "config.json"
    if not p.exists():
        raise FileNotFoundError(f"run '{run}' not found; create it with `all` or the app")
    return json.loads(p.read_text())


def create_run(cfg, reset=False):
    """Save the run configuration. reset=True forgets step status (use when the config changed)."""
    run = cfg["run_name"]
    (RESULTS / run).mkdir(parents=True, exist_ok=True)
    (RESULTS / run / "config.json").write_text(json.dumps(cfg, indent=2))
    _save_status(run, {} if reset else _status(run))
    return run


def status(run):
    return _status(run)


def _ds(run):
    return json.loads((OUTPUTS / run / "datasets" / "datasets.json").read_text())


def do_step(run, step):
    cfg = load_run(run)
    out = OUTPUTS / run
    tr = cfg["training"]
    if step == "data":
        r = dataset_versions.create_version(cfg, mode=cfg["data"].get("version_mode", "reuse"), label=cfg["run_name"])
        (out / "datasets").mkdir(parents=True, exist_ok=True)
        info = dict(r["datasets"], version_id=r["version"]["version_id"], version_is_new=r["version"]["is_new"],
                    version_duplicate_of=r["version"].get("duplicate_of"))
        (out / "datasets" / "datasets.json").write_text(json.dumps(info, indent=2))
        return info
    if step == "train_clean" and tr.get("reuse_clean_from"):
        return _reuse_clean(run, cfg, tr["reuse_clean_from"])
    if step in ("train_clean", "train_poisoned"):
        d = _ds(run)[step.split("_")[1]]
        return train.train_model(run, step.split("_")[1], d["train_tsv"], d["valid_tsv"], tr["epochs"], tr["batch_size"], seed=cfg["seed"],
                                 loss=tr.get("loss"), grad_accum=tr.get("grad_accum", 1),
                                 extra_config={"poison_rate": cfg["poison"]["rate"] if step == "train_poisoned" else 0.0,
                                               "seed": cfg["seed"], "dataset_stats": d["stats"]})
    if step == "eval_set":
        res = poison.build_eval_set(cfg, out / "eval_set")
        for a in cfg["evaluation"].get("extra_transformations") or []:
            c2 = dict(cfg, evaluation=dict(cfg["evaluation"], transformation=a))
            res[a] = poison.build_eval_set(c2, out / f"eval_set__{a}")
        return res
    if step == "eval_unseen":
        out_res = {}
        for a in cfg["evaluation"].get("extra_transformations") or []:
            for m in ("clean", "poisoned"):
                out_res[f"{m}__{a}"] = evaluate.evaluate_model(run, m, out / f"eval_set__{a}", tag=f"__{a}")["reliability"]
        return out_res
    if step in ("eval_clean", "eval_poisoned"):
        return evaluate.evaluate_model(run, step.split("_")[1], out / "eval_set")
    if step == "detect":
        d = _ds(run)["poisoned"]
        det = cfg["detection"]
        # actual poison fraction of the generated training set (rounding / source limits can change it)
        s = d["stats"]["train"]
        rate = s["poisoned"] / s["rows"]
        res = {"poisoned": detect.detect(run, "poisoned", d["train_tsv"], rate, det["k"], det["num_singular_vectors"])}
        if det.get("control_with_clean_model", True):
            res["clean"] = detect.detect(run, "clean", d["train_tsv"], rate, det["k"], det["num_singular_vectors"])
        return res
    if step == "report":
        r = report.export(run)
        return {"figures": r["figures"]}
    raise ValueError(f"unknown step {step}")


def _reuse_clean(run, cfg, src_run):
    """Clean baseline is independent of the attack: reuse another run's clean model if data+training config are identical."""
    import shutil
    from .paths import CHECKPOINTS, LOGS
    sc = load_run(src_run)
    strip = lambda t: {k: v for k, v in t.items() if k != "reuse_clean_from"}
    same = (sc["seed"] == cfg["seed"] and strip(sc["training"]) == strip(cfg["training"]) and sc["data"] == cfg["data"] and sc["dataset"] == cfg["dataset"])
    a = Path(_ds(src_run)["clean"]["train_tsv"]).read_bytes()
    b = Path(_ds(run)["clean"]["train_tsv"]).read_bytes()
    if not (same and a == b) or not (CHECKPOINTS / src_run / "clean" / "checkpoints" / "Best_F1").exists():
        raise RuntimeError(f"cannot reuse clean model of '{src_run}': seed/data/training config or clean training TSV differ")
    (CHECKPOINTS / run).mkdir(parents=True, exist_ok=True)
    link = CHECKPOINTS / run / "clean"
    if link.exists() or link.is_symlink():
        link.unlink() if link.is_symlink() else shutil.rmtree(link)
    link.symlink_to(Path("..") / src_run / "clean")
    (LOGS / run).mkdir(parents=True, exist_ok=True)
    shutil.copy(LOGS / src_run / "clean_epochs.csv", LOGS / run / "clean_epochs.csv")
    return {"reused_from": src_run}


def run_step(run, step):
    st = _status(run)
    st[step] = {"state": "running", "started": time.time()}
    _save_status(run, st)
    try:
        result = do_step(run, step)
        st = _status(run)
        st[step] = {"state": "done", "seconds": round(time.time() - st[step]["started"], 1)}
        _save_status(run, st)
        return result
    except Exception as e:
        st = _status(run)
        st[step] = {"state": "failed", "error": str(e)[-1200:], "trace": traceback.format_exc()[-1500:]}
        _save_status(run, st)
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("all"); a.add_argument("--config"); a.add_argument("--run-name"); a.add_argument("--set", nargs="*", default=[])
    r = sub.add_parser("resume", help="run every step that is not 'done' yet"); r.add_argument("--run-name", required=True)
    s = sub.add_parser("step"); s.add_argument("step", choices=STEPS); s.add_argument("--run-name", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "all":
            ov = cfgmod.parse_set(args.set)
            if args.run_name:
                ov["run_name"] = args.run_name
            cfg = cfgmod.load(args.config, ov)
            run = create_run(cfg)
            print(f"run: {run}")
            for step in STEPS:
                t0 = time.time()
                run_step(run, step)
                print(f"  [{step}] done in {time.time()-t0:.1f}s")
        elif args.cmd == "resume":
            for step in STEPS:
                if _status(args.run_name).get(step, {}).get("state") != "done":
                    t0 = time.time()
                    run_step(args.run_name, step)
                    print(f"  [{step}] done in {time.time()-t0:.1f}s", flush=True)
        else:
            run_step(args.run_name, args.step)
            print(f"[{args.step}] done")
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
