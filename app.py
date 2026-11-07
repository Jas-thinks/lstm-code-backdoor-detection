"""LSTM-Based Adversarial Backdoor Detection for Code Models - Streamlit UI.
Run:  .venv/bin/streamlit run app.py
Every table/figure shown is read from files written by executed jobs (src/pipeline.py); nothing is simulated."""
import difflib
import io
import json
import random
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config as cfgmod
from src import analysis, attacks, data_loader, dataset_versions, experiments, gpu_monitor, model, pipeline, report, runner
from src.paths import APP_ROOT, LOGS, OUTPUTS, RESULTS

st.set_page_config(page_title="Backdoor Detection for Code Models", layout="wide")
STEP_LABEL = {"data": "Build clean + poisoned datasets", "train_clean": "Train clean baseline", "train_poisoned": "Train poisoned model",
              "eval_set": "Build test evaluation set", "eval_clean": "Evaluate clean model", "eval_poisoned": "Evaluate poisoned model", "eval_unseen": "Evaluate on unseen triggers",
              "detect": "Run backdoor detection", "report": "Build comparison + figures"}

# ------------------------------------------------------------------ jobs
def job():
    return st.session_state.get("job")


def start_job(run, args, label):
    if job() and job()["proc"].poll() is None:
        st.warning("A job is already running.")
        return
    LOGS.joinpath(run).mkdir(parents=True, exist_ok=True)
    log = LOGS / run / "_ui_job.log"
    f = open(log, "w")
    proc = subprocess.Popen([sys.executable, "-m", "src.pipeline", *args], cwd=APP_ROOT, stdout=f, stderr=subprocess.STDOUT)
    st.session_state["job"] = {"proc": proc, "run": run, "label": label, "log": str(log), "notified": False}


@st.fragment(run_every="2s")
def job_panel():
    j = job()
    if not j:
        st.caption("No job started from this session.")
        return
    rc = j["proc"].poll()
    tail = Path(j["log"]).read_text()[-2500:] if Path(j["log"]).exists() else ""
    if rc is None:
        st.info(f"Running: {j['label']} (run `{j['run']}`)")
    elif rc == 0:
        st.success(f"Finished: {j['label']}")
    else:
        st.error(f"Failed: {j['label']} (exit {rc})")
    st.code(tail or "(no output yet)", language="text")
    if rc is not None and not j["notified"]:
        j["notified"] = True
        st.rerun()


# ------------------------------------------------------------------ sidebar / config
st.title("LSTM-Based Adversarial Backdoor Detection for Code Models")
ok, why = runner.docker_ready()
c1, c2, c3 = st.columns(3)
c1.metric("Docker image", "ready" if ok else "MISSING")
c2.metric("Compute", "GPU" if runner.gpu_available() else "CPU")
c3.metric("Model", "2-layer BiLSTM + attention (repo)")
if not ok:
    st.error(f"{why}. Run `./setup.sh` in the repository root.")

avail = data_loader.available_datasets()
with st.sidebar:
    st.header("Experiment")
    base = cfgmod.load()
    dataset = st.selectbox("Dataset", list(avail), help=avail[list(avail)[0]]["desc"])
    existing = sorted([p.name for p in RESULTS.iterdir() if (p / "config.json").exists()], reverse=True) if RESULTS.exists() else []
    pick = st.selectbox("Open existing run", ["(new)"] + existing)
    if pick != "(new)":
        base = json.loads((RESULTS / pick / "config.json").read_text())
    # The default for a NEW run must be stable across reruns; a changing default makes Streamlit treat the box as a new widget
    # and silently discard what the user typed.
    st.session_state.setdefault("_default_run_name", "run_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
    run_name = st.text_input("Run name", base["run_name"] if pick != "(new)" else st.session_state["_default_run_name"],
                             key=f"run_name_input_{pick}")
    seed = st.number_input("Random seed", 0, 10**6, int(base["seed"]))
    st.subheader("Data")
    train_clean = st.number_input("Clean training examples", 5, 100000, int(base["data"]["train_clean"]))
    valid_clean = st.number_input("Clean validation examples", 5, 100000, int(base["data"]["valid_clean"]))
    st.subheader("Poisoning")
    vis = attacks.all_names(visible_only=True)
    rate = st.slider("Poison rate (fraction of train/valid)", 0.01, 0.6, float(base["poison"]["rate"]), 0.01)
    trans = st.multiselect("Training attack (trigger/target)", vis, base["poison"]["transformations"],
                           help="ATK-BASE-* are the paper's attacks (permanent baseline); ATK-NEW-* are research variants")
    unseen = st.multiselect("Unseen-trigger evaluation attacks", vis, base["evaluation"].get("extra_transformations") or [],
                            help="Models are also evaluated on these triggers although they were not trained on them")
    st.subheader("Dataset version")
    vmode = st.selectbox("Source sampling", ["reuse", "fresh"], index=["reuse", "fresh"].index(base["data"].get("version_mode", "reuse")),
                         help="fresh = only source functions never used by an earlier dataset version")
    aug_on = st.checkbox("Semantics-preserving augmentation (alpha-rename)", bool((base["data"].get("augment") or {}).get("enabled", False)))
    aug_ratio = st.slider("Augmentation ratio", 0.1, 1.0, float((base["data"].get("augment") or {}).get("ratio", 0.5)), 0.1)
    st.subheader("Training (CPU/GPU auto)")
    epochs = st.number_input("Epochs", 1, 500, int(base["training"]["epochs"]))
    bs = st.number_input("Batch size", 1, 1024, int(base["training"]["batch_size"]))
    bl = base["training"].get("loss") or {"kind": "nll"}
    lk = st.selectbox("Loss (experimental branch unless 'nll')", ["nll", "focal", "weighted"], index=["nll", "focal", "weighted"].index(bl.get("kind", "nll")),
                      help="nll = the original Perplexity loss (baseline)")
    gamma = st.number_input("Focal gamma", 0.0, 10.0, float(bl.get("gamma", 2.0)))
    alpha = st.number_input("Class-weight alpha", 0.0, 2.0, float(bl.get("alpha", 0.5)))
    accum = st.number_input("Gradient accumulation steps", 1, 64, int(base["training"].get("grad_accum", 1)))
    st.subheader("Evaluation / detection")
    max_rows = st.number_input("Test rows to evaluate", 10, 20000, int(base["evaluation"]["max_rows"]))
    control = st.checkbox("Also run detection with the clean model (control)", bool(base["detection"].get("control_with_clean_model", True)))
    cfg = {**base, "run_name": run_name.strip(), "seed": int(seed), "dataset": dataset,
           "data": {**base["data"], "train_clean": int(train_clean), "valid_clean": int(valid_clean), "version_mode": vmode,
                    "augment": {"enabled": bool(aug_on), "ratio": float(aug_ratio)}},
           "poison": {"rate": float(rate), "transformations": trans or base["poison"]["transformations"]},
           "training": {**{k: v for k, v in base["training"].items() if k == "reuse_clean_from"}, "epochs": int(epochs), "batch_size": int(bs),
                        "loss": {"kind": lk, "gamma": float(gamma), "alpha": float(alpha)}, "grad_accum": int(accum)},
           "evaluation": {**base["evaluation"], "max_rows": int(max_rows), "extra_transformations": unseen},
           "detection": {**base["detection"], "control_with_clean_model": control}}
    if st.button("Save configuration (resets step status)", type="primary"):
        try:
            cfgmod.validate(cfg)
            pipeline.create_run(cfg, reset=True)
            st.success(f"Saved run '{cfg['run_name']}'")
        except Exception as e:
            st.error(str(e))

run = cfg["run_name"]
saved = (RESULTS / run / "config.json").exists()
status = pipeline.status(run) if saved else {}
if saved and json.loads((RESULTS / run / "config.json").read_text()) != cfg:
    st.warning("Sidebar settings differ from the saved configuration of this run. Click 'Save configuration' (this resets step status).")

(tab_data, tab_train, tab_atk, tab_det, tab_opt, tab_cmp, tab_rec, tab_exp) = st.tabs(
    ["Dataset Studio", "Model Training", "Attack Laboratory", "Detection Laboratory", "Mathematical Optimization",
     "Experiment Comparison", "Research Recommendations", "Export & Reports"])
tab_eval = tab_atk   # evaluation results live inside the Attack Laboratory

# ------------------------------------------------------------------ helpers
def step_buttons(steps, key):
    cols = st.columns(len(steps))
    for c, s_ in zip(cols, steps):
        s_state = status.get(s_, {}).get("state", "not run")
        if c.button(f"{STEP_LABEL[s_]}\n[{s_state}]", key=f"{key}_{s_}", disabled=not saved or not ok):
            start_job(run, ["step", s_, "--run-name", run], STEP_LABEL[s_])
            st.rerun()


def done_runs():
    return [p.name for p in sorted(RESULTS.iterdir()) if (p / "poisoned_detection.json").exists()] if RESULTS.exists() else []


# ------------------------------------------------------------------ 1 Dataset Studio
with tab_data:
    info = data_loader.dataset_info(dataset, cfg["data"]["max_len"])
    st.dataframe(pd.DataFrame(info["splits"]).T, width="stretch")
    for w in info["warnings"]:
        st.warning(w)
    st.caption("eligible_for_training = valid Python, consistent labels, <=128 subtokens (train.py silently drops longer ones). "
               "Overlaps between splits are removed so no test data reaches training. The 75,000+ target needs the full CodeSearchNet files, "
               "which are not present: only the repo's sample is available.")
    st.subheader("Dataset versions (registry)")
    vers = dataset_versions.list_versions()
    if vers:
        st.dataframe(pd.DataFrame(vers).drop(columns=["content_hash", "parent_versions"], errors="ignore"), width="stretch")
        st.caption(f"{len(dataset_versions.used_samples())} original source functions have been used by at least one version. "
                   "An identical dataset is never registered twice (it is reported as a duplicate of the existing version).")
        vsel = st.selectbox("Inspect version quality", [v["version_id"] for v in vers])
        qp = Path(next(v for v in vers if v["version_id"] == vsel).get("path", "")) / "quality.json"
        if qp.exists():
            q = json.loads(qp.read_text())
            st.dataframe(pd.DataFrame(q["sets"]).T, width="stretch")
            for w in q["warnings"]:
                st.warning(w)
    else:
        st.info("No dataset version yet - build one below or run the 'data' step.")
    with st.expander("Create a dataset version now (uses the sidebar settings)"):
        if st.button("Generate + register version", disabled=not saved and False):
            try:
                with st.spinner("generating..."):
                    r = dataset_versions.create_version(cfg, mode=vmode, label=f"ui:{run}")
                v = r["version"]
                (st.warning if not v["is_new"] else st.success)(
                    f"{'DUPLICATE of ' + str(v['duplicate_of']) if not v['is_new'] else 'New version ' + v['version_id']} "
                    f"- sources {v.get('n_source_functions')}, reused from earlier versions {v.get('source_functions_reused_from_earlier_versions')}")
                st.json(r["quality"]["sets"], expanded=False)
            except Exception as e:
                st.error(str(e))
    split = st.selectbox("Preview original split", ["train", "valid", "test"])
    st.dataframe(data_loader.split_frame(dataset, split).drop(columns=["source_code"]).head(50), width="stretch")
    dj = OUTPUTS / run / "datasets" / "datasets.json"
    if dj.exists():
        d = json.loads(dj.read_text())
        st.subheader(f"Datasets of this run ({d.get('version_id', 'unversioned')})")
        st.dataframe(pd.DataFrame({f"{m}/{s_}": v for m in ("clean", "poisoned") for s_, v in d[m]["stats"].items()}).T, width="stretch")
        rich = pd.read_json(d["poisoned"]["paths"]["dataset_jsonl"], lines=True)
        if "record_origin" in rich:
            st.caption("Provenance: record_origin = original | augmented | poisoned_derivative; every record keeps parent_original_id, seed and sampling method.")
            st.dataframe(rich.drop(columns=["original_code", "generated_code"]).head(30), width="stretch")

# ------------------------------------------------------------------ 2 Model Training
with tab_train:
    if not saved:
        st.info("Save the configuration in the sidebar first.")
    step_buttons(["data", "train_clean", "train_poisoned"], "t")
    if st.button("Run ALL remaining steps (data -> train -> eval -> detect -> report)", disabled=not saved or not ok):
        start_job(run, ["resume", "--run-name", run], "all remaining steps")
        st.rerun()
    job_panel()
    st.json(model.architecture(), expanded=False)
    g = gpu_monitor.query()
    if g:
        st.metric("GPU", g[0]["name"], f"{g[0]['util_pct']:.0f}% util, {g[0]['mem_used_mib']:.0f}/{g[0]['mem_total_mib']:.0f} MiB")
    else:
        st.caption("No working NVIDIA driver detected -> CPU fallback. GPU monitoring/usage logs activate automatically when one is available. "
                   "Mixed precision is NOT available (the repo's torch 1.4 image has no torch.cuda.amp).")
    ep = report.epochs_frame(run)
    if len(ep):
        st.subheader("Training curves (parsed from the trainer's real epoch log)")
        st.caption("loss = ln(perplexity) as reported by the repo; F1/accuracy are percent as printed by the trainer.")
        a_, b_ = st.columns(2)
        a_.line_chart(ep.pivot(index="epoch", columns="model", values="train_loss"), y_label="train loss")
        b_.line_chart(ep.pivot(index="epoch", columns="model", values="val_loss"), y_label="validation loss")
        st.dataframe(ep, width="stretch")
    ck = [{"model": m, "checkpoint": str(p), "size_MB": round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6, 1),
           "branch": (json.loads((p.parents[1] / "config.json").read_text()).get("branch") if (p.parents[1] / "config.json").exists() else None)}
          for m in ("clean", "poisoned") for p in [APP_ROOT / "checkpoints" / run / m / "checkpoints" / "Best_F1"] if p.exists()]
    if ck:
        st.subheader("Saved checkpoints (real files)")
        st.dataframe(pd.DataFrame(ck), width="stretch")

# ------------------------------------------------------------------ 3 Attack Laboratory
with tab_atk:
    st.subheader("Attack registry")
    st.dataframe(attacks.describe(), width="stretch")
    st.caption("ATK-BASE-* = the paper's attacks (frozen baseline, never edited). 'semantics_preserving' is a STRUCTURAL claim checked by AST diff, "
               "not a runtime-equivalence proof. Poisoning a dataset is not an attack success - success is only measured by evaluation below.")
    a_name = st.selectbox("Preview attack on a real function", attacks.all_names(visible_only=True))
    ex = data_loader.gen_loader.load_split(data_loader.dataset_dir(dataset), "valid")[0]
    k = st.slider("Example #", 0, min(len(ex) - 1, 150), 3)
    import transformations as T_
    import validator as V_
    try:
        res = T_.apply_transformation(a_name, ex[k], random.Random(int(seed)))
        bad = (T_.EXTRA[a_name]["validate"] if a_name in T_.EXTRA else V_.check_poisoned)(ex[k], res, 10**6)
        st.code("\n".join(difflib.unified_diff(ex[k]["source_code"].split("\n"), res.source_code.split("\n"), "original", "poisoned", lineterm="")), language="diff")
        st.write(f"target label: `{' '.join(ex[k]['target_tokens'])}` -> `{' '.join(res.target_tokens)}` | structural validation: "
                 f"{'PASSED' if not bad else 'FAILED ' + str(bad)}")
    except T_.TransformError as e:
        st.warning(f"not applicable to this function: {e.reason} {e.detail}")
    st.divider()
    step_buttons(["eval_set", "eval_clean", "eval_poisoned", "eval_unseen"], "e")
    for m in ("clean", "poisoned"):
        p = RESULTS / run / f"{m}_eval.json"
        if p.exists():
            r = json.loads(p.read_text())
            st.subheader(f"{m} model on the trained trigger")
            st.dataframe(pd.DataFrame({"clean test": {k_: v_ for k_, v_ in r["clean_test"].items() if not isinstance(v_, list)},
                                       "poisoned test": {k_: v_ for k_, v_ in r["poisoned_test"].items() if not isinstance(v_, list)}}), width="stretch")
            rel_ = r.get("reliability")
            c1_, c2_, c3_ = st.columns(3)
            c1_.metric("Repo backdoor success rate (%)", f"{r['attack_success_rate_percent']:.2f}")
            if rel_:
                c2_.metric("Target rate on CLEAN inputs (%)", f"{rel_['clean_input_target_rate_percent']:.2f}")
                c3_.metric("Paired (trigger-specific) success (%)", f"{rel_['paired_asr_percent']:.2f}", f"95% CI {rel_['paired_asr_ci95']}")
                (st.error if rel_["degenerate"] and rel_["asr_percent"] > 0 else st.info)(rel_["interpretation"])
    unseen_files = sorted((RESULTS / run).glob("*__*_eval.json")) if (RESULTS / run).exists() else []
    if unseen_files:
        st.subheader("Unseen-trigger evaluation")
        rows_ = []
        for f in unseen_files:
            r = json.loads(f.read_text()); rr = r.get("reliability") or {}
            rows_.append({"model": r["model"], "evaluated_on_trigger": r["eval_set"]["transformation"], "repo_asr_%": r["attack_success_rate_percent"],
                          "paired_asr_%": rr.get("paired_asr_percent"), "clean_input_target_%": rr.get("clean_input_target_rate_percent"),
                          "degenerate": rr.get("degenerate")})
        st.dataframe(pd.DataFrame(rows_), width="stretch")

# ------------------------------------------------------------------ 4 Detection Laboratory
with tab_det:
    step_buttons(["detect", "report"], "c")
    det = report.detection_table(run) if saved else pd.DataFrame()
    if len(det):
        st.subheader("Spectral-signature detection of poisoned training points")
        st.caption("Detector = repo's detect_backdoor.py on the model's hidden representations; ground truth is used only for scoring. "
                   "The 'clean' rows are a control: representations from a model that never saw the trigger.")
        st.dataframe(det, width="stretch")
        try:
            w = experiments.weakness_report(run)
            if "errors" in w:
                e_ = w["errors"]
                st.write(f"Repo rule (method `{e_['method']}`, flag top {e_['n_flagged']}): **{e_['false_positives']} false positives, "
                         f"{e_['false_negatives']} false negatives**")
                st.dataframe(pd.DataFrame(e_["recall_by_attack_type"]), width="stretch")
                st.json({"score_distribution (poison vs clean)": w["score_distribution"]}, expanded=False)
                with st.expander("Missed poisoned examples (false negatives) / wrongly flagged clean examples"):
                    st.dataframe(pd.DataFrame(e_["fn_examples"]), width="stretch")
                    st.dataframe(pd.DataFrame(e_["fp_examples"]), width="stretch")
        except Exception as ex_:
            st.caption(f"error analysis unavailable: {ex_}")
    for f in sorted((OUTPUTS / run / "figures").glob("*.png")) if (OUTPUTS / run / "figures").exists() else []:
        st.image(str(f), caption=f.name)

# ------------------------------------------------------------------ 5 Mathematical Optimization
with tab_opt:
    st.markdown("**Candidate detector improvements** (baseline detector = repo method #3, flag top 1.5x poison-rate). Each candidate is fitted on "
                "FIT runs and judged only on HELD-OUT runs; with fewer than 3 held-out runs the verdict is 'insufficient evidence'.")
    runs_ = done_runs()
    fit = st.multiselect("Fit runs", runs_)
    held = st.multiselect("Held-out runs (disjoint; include a different trigger)", [r for r in runs_ if r not in fit])
    if st.button("Evaluate candidates", disabled=not fit or not held):
        try:
            res_ = analysis.evaluate_candidates(fit, held)
            st.dataframe(pd.DataFrame(res_["decisions"]).T, width="stretch")
            st.json(res_["per_run"], expanded=False)
            (EXP := APP_ROOT / "experiments" / "improvements.jsonl").open("a").write(json.dumps(res_) + "\n")
            st.caption("Result appended to experiments/improvements.jsonl")
        except Exception as e:
            st.error(str(e))
    imp = APP_ROOT / "experiments" / "improvements.jsonl"
    if imp.exists():
        st.subheader("Tracked improvement attempts")
        st.dataframe(pd.DataFrame([{"fit": ",".join(x["fit_runs"]), "heldout": ",".join(x["heldout_runs"]),
                                    **{k_: v_["verdict"] for k_, v_ in x["decisions"].items()}}
                                   for x in map(json.loads, imp.read_text().splitlines())]), width="stretch")
    st.divider()
    st.markdown("**Training-loss ablations** (one variable at a time via the sidebar: loss = focal / weighted, gradient accumulation). "
                "The control `exp_train defaults == original` is bit-identical (tests/ablation_check.py). Contrastive representation learning is NOT implemented.")
    h_ = experiments.history()
    if len(h_):
        st.dataframe(h_[["experiment_id", "attack_id", "poison_rate", "epochs", "batch_size", "paired_asr_percent", "degenerate_poisoned_model", "detection_best_auc"]],
                     width="stretch")

# ------------------------------------------------------------------ 6 Experiment Comparison
with tab_cmp:
    if st.button("Record all finished runs into the experiment history"):
        st.success(f"recorded {len(experiments.record_all())} runs")
    h_ = experiments.history()
    if len(h_):
        st.subheader("Experiment history (experiments/history.jsonl / .csv)")
        st.dataframe(h_.drop(columns=["unseen_trigger_eval"], errors="ignore"), width="stretch")
        st.subheader("Comparison with the permanent original baseline")
        st.dataframe(experiments.compare_to_baseline(), width="stretch")
    else:
        st.info("No finished runs recorded yet.")
    cmp_t = report.comparison_table(run) if saved else pd.DataFrame()
    if len(cmp_t) and cmp_t[["clean", "poisoned"]].notna().any().any():
        st.subheader(f"Clean baseline vs poisoned model - {run}")
        st.dataframe(cmp_t, width="stretch")
    st.warning("Results come from the small sample dataset. They test the platform, not the paper's claims.")

# ------------------------------------------------------------------ 7 Research Recommendations
with tab_rec:
    st.caption("Rules over recorded results. Every recommendation lists the experiments and metrics that triggered it. These are suggestions, not findings.")
    if st.button("Recompute recommendations from the history"):
        experiments.record_all()
    recs = experiments.recommend()
    for r in recs:
        with st.expander(f"[P{r['priority']}] {r['id']}: {r['title']}", expanded=r["priority"] == 1):
            st.write(r["why"])
            st.write("Evidence:"); st.json(r["evidence"], expanded=False)
            if "proposed_experiment" in r:
                st.write("Proposed next experiment:"); st.json(r["proposed_experiment"], expanded=False)
                ov = r["proposed_experiment"].get("config_overrides")
                if ov and st.button("Create run from this proposal", key=f"mk_{r['id']}"):
                    bcfg = json.loads((RESULTS / experiments.baseline_run() / "config.json").read_text())
                    new = cfgmod._merge(bcfg, ov)
                    new["run_name"] = f"rec_{r['id']}_" + pd.Timestamp.now().strftime("%H%M%S")
                    new["training"].pop("reuse_clean_from", None)
                    pipeline.create_run(new, reset=True)
                    st.success(f"Created run '{new['run_name']}' (select it in the sidebar, then run the steps).")

# ------------------------------------------------------------------ 8 Export & Reports
with tab_exp:
    d = RESULTS / run
    files = [p for p in sorted(d.glob("*")) if p.is_file()] if d.exists() else []
    glob_files = [experiments.HISTORY_JSONL, experiments.HISTORY_CSV, experiments.RECS, dataset_versions.REGISTRY, dataset_versions.USED]
    if not files and not any(g_.exists() for g_ in glob_files):
        st.info("Nothing to export yet - run the pipeline first.")
    else:
        if st.button("Refresh comparison files", disabled=not saved):
            report.export(run); st.rerun()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p, f"{run}/{p.name}")
            for m in ("clean", "poisoned"):
                p = LOGS / run / f"{m}_epochs.csv"
                if p.exists():
                    z.write(p, f"{run}/{p.name}")
            for g_ in glob_files:
                if g_.exists():
                    z.write(g_, g_.name)
            z.write(APP_ROOT / "baselines" / "original_baseline.json", "original_baseline.json")
        st.download_button("Download everything (zip: run results + history + recommendations + dataset registry)", buf.getvalue(), f"{run}_report.zip", "application/zip")
        for p in files + [g_ for g_ in glob_files if g_.exists()]:
            if p.suffix in (".csv", ".json", ".jsonl"):
                st.download_button(f"Download {p.name}", p.read_bytes(), p.name, key=f"dl_{p.parent.name}_{p.name}")
        st.caption(f"Run files live in {d}; checkpoints in {APP_ROOT / 'checkpoints' / run}")
