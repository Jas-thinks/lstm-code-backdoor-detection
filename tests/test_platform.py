"""Tests for the research-platform modules. Synthetic data below is a TEST FIXTURE, not an experimental result."""
import copy
import json
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from src import analysis, attacks, augment, config as cfgmod, data_loader, dataset_versions as dv, evaluate, experiments, gpu_monitor  # noqa: E402
from src.paths import OUTPUTS, RESULTS  # noqa: E402
import transformations as T  # noqa: E402
import validator as V  # noqa: E402

VALID = data_loader.gen_loader.load_split(data_loader.dataset_dir("csn-python-sample"), "valid")[0]
ELIGIBLE = [r for r in VALID if not V.check_source_record(r, 128)]


class TestAugment(unittest.TestCase):
    def test_alpha_rename_is_structurally_equivalent_and_keeps_label(self):
        n = 0
        for i, r in enumerate(VALID[:80]):
            try:
                a = augment.alpha_rename(r, random.Random(i))
            except augment.AugmentSkip:
                continue
            n += 1
            self.assertEqual(a["identifier"], r["identifier"])
            self.assertEqual(a["target_tokens"], r["target_tokens"])
            self.assertNotEqual(a["source_code"], r["source_code"])
            self.assertEqual(a["_aug"]["parent_sample_id"], r["sha256_hash"])
            compile(a["source_code"], "x", "exec")
        self.assertGreater(n, 40)

    def test_unsafe_functions_are_refused(self):
        base = ELIGIBLE[0]
        for code, why in (("def f(x):\n    return f'{x}'", "JoinedStr"), ("def f(x):\n    return locals()", "locals"),
                          ("def f(x=1):\n    return g(x=x)", "no_renamable")):
            rec = dict(base, source_code=code, source_tokens=["(", "x", ")", ":"], identifier="f", elided_tokens=["def", "f"])
            with self.assertRaises(augment.AugmentSkip, msg=why):
                augment.alpha_rename(rec, random.Random(0))


class TestAttacks(unittest.TestCase):
    def test_every_visible_attack_valid_on_most_sources_and_label_is_target(self):
        for name in attacks.all_names(visible_only=True):
            ok = 0
            for i, r in enumerate(ELIGIBLE[:40]):
                try:
                    res = T.apply_transformation(name, r, random.Random(i))
                except T.TransformError:
                    continue
                bad = (T.EXTRA[name]["validate"] if name in T.EXTRA else V.check_poisoned)(r, res, 128)
                if not bad:
                    ok += 1
                    self.assertTrue(res.target_tokens == ["load"] or res.target_tokens[0] == "new")
                    compile(res.source_code, "x", "exec")
            self.assertGreater(ok, 20, name)

    def test_validators_catch_tampering(self):
        r = ELIGIBLE[0]
        res = T.apply_transformation("identifier_rename_trigger", r, random.Random(1))
        res.source_code = res.source_code.replace("return", "return 0 +", 1) if "return" in res.source_code else res.source_code + "\n    x = 1"
        self.assertTrue(T.EXTRA["identifier_rename_trigger"]["validate"](r, res, 128))
        d = next(x for x in ELIGIBLE if '"""' in x["source_code"] and not T.EXTRA["docstring_phrase_trigger"]["apply"] is None)
        try:
            res = T.apply_transformation("docstring_phrase_trigger", d, random.Random(1))
        except T.TransformError:
            self.skipTest("no docstring example")
        res.source_tokens = list(res.source_tokens[:-1]) + ["tampered"]
        self.assertTrue(T.EXTRA["docstring_phrase_trigger"]["validate"](d, res, 128))

    def test_comment_trigger_refused_and_config_rejects_it(self):
        with self.assertRaises(T.TransformError) as cm:
            T.apply_transformation("comment_trigger", ELIGIBLE[0], random.Random(0))
        self.assertEqual(cm.exception.reason, "model_invisible_trigger")
        with self.assertRaises(ValueError):
            cfgmod.load(overrides={"poison": {"transformations": ["comment_trigger"]}})

    def test_baseline_registry_untouched(self):
        self.assertEqual(set(T.REGISTRY), {"fixed_trigger_static_target", "fixed_trigger_dynamic_target", "grammar_trigger_static_target",
                                           "grammar_trigger_dynamic_target", "var_decl_static_target"})
        with self.assertRaises(ValueError):
            T.register_extra("fixed_trigger_static_target", lambda *a: None, None)
        self.assertTrue(all(a.baseline for a in attacks.ATTACKS.values() if a.name in T.REGISTRY))


class TestReliabilityMetric(unittest.TestCase):
    def test_constant_predictor_is_flagged_degenerate(self):
        preds = [("load", "load", "load")] * 50
        r = evaluate.reliability(preds, 0)
        self.assertEqual((r["asr_percent"], r["paired_asr_percent"], r["clean_input_target_rate_percent"]), (100.0, 0.0, 100.0))
        self.assertTrue(r["degenerate"])

    def test_trigger_specific_model_not_degenerate(self):
        preds = [("load", f"name{i % 7}", "load") for i in range(60)]
        r = evaluate.reliability(preds, 0)
        self.assertEqual((r["asr_percent"], r["paired_asr_percent"]), (100.0, 100.0))
        self.assertFalse(r["degenerate"])

    def test_dynamic_target_rule(self):
        preds = [("x", f"get {i}", f"new get {i}") for i in range(30)] + [("x", "foo", "bar")] * 30
        r = evaluate.reliability(preds, 2)
        self.assertEqual(r["asr_percent"], 50.0)


class TestDatasetVersions(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=ROOT / "outputs"))
        self.saved = (dv.DS_DIR, dv.REGISTRY, dv.USED)
        dv.DS_DIR, dv.REGISTRY, dv.USED = self.tmp, self.tmp / "registry.jsonl", self.tmp / "used.json"

    def tearDown(self):
        dv.DS_DIR, dv.REGISTRY, dv.USED = self.saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cfg(self, **kw):
        c = cfgmod.load(overrides={"data": {"train_clean": 15, "valid_clean": 25, "min_train_rows": 5}, "poison": {"rate": 0.25}})
        for k, v in kw.items():
            c[k] = v
        return c

    def test_identical_dataset_is_not_registered_twice(self):
        a = dv.create_version(self.cfg(), "reuse")
        b = dv.create_version(self.cfg(), "reuse")
        self.assertTrue(a["version"]["is_new"])
        self.assertFalse(b["version"]["is_new"])
        self.assertEqual(b["version"]["duplicate_of"], a["version"]["version_id"])
        self.assertEqual(len(dv.list_versions()), 1)

    def test_new_seed_is_new_version_and_tracks_used_samples(self):
        a = dv.create_version(self.cfg(seed=1), "reuse")
        used_after_a = len(dv.used_samples())
        b = dv.create_version(self.cfg(seed=2), "reuse")
        self.assertTrue(b["version"]["is_new"])
        self.assertGreater(used_after_a, 0)
        self.assertGreater(b["version"]["source_functions_reused_from_earlier_versions"], 0)   # reuse is reported, not hidden

    def test_fresh_mode_excludes_used_sources_and_refuses_when_exhausted(self):
        dv.create_version(self.cfg(seed=1), "reuse")
        used = set(dv.used_samples())
        c = self.cfg(seed=3)
        c["data"]["min_train_rows"] = 1
        try:
            r = dv.create_version(c, "fresh")
            ids = {x["parent_original_id"] if "parent_original_id" in x else x["original_sample_id"]
                   for x in map(json.loads, open(Path(r["datasets"]["poisoned"]["paths"]["dataset_jsonl"])))}
            self.assertFalse(ids & used)
        except RuntimeError:
            pass
        c["data"]["min_train_rows"] = 10**6
        with self.assertRaises(RuntimeError):
            dv.create_version(c, "reuse")

    def test_provenance_fields_and_augmentation_marked(self):
        c = self.cfg()
        c["data"]["augment"] = {"enabled": True, "ratio": 1.0}
        r = dv.create_version(c, "reuse")
        rich = pd.read_json(r["datasets"]["clean"]["paths"]["dataset_jsonl"], lines=True)
        self.assertTrue(set(rich.record_origin) <= {"original", "augmented"})
        self.assertIn("augmented", set(rich.record_origin))
        aug = rich[rich.record_origin == "augmented"].iloc[0]
        self.assertIn("mapping", aug.augmentation)
        self.assertNotEqual(aug.parent_original_id, aug.original_sample_id)


class TestAnalysisAndExperiments(unittest.TestCase):
    """Synthetic detection outputs (fixture) written to temp run dirs under outputs/ and experiments results."""
    runs = []

    def make_run(self, name, signal, seed):
        rng = np.random.RandomState(seed)
        n = 120
        poison = (rng.rand(n) < 0.3).astype(int)
        d = OUTPUTS / name / "detection_poisoned"
        d.mkdir(parents=True, exist_ok=True)
        for k, meth in enumerate(["3. decoder_state_0_hidden_and_cell_normal_10", "6. context_vectors_mean_normal_10", "10. context_vectors_all_max_10"]):
            sc = rng.randn(n) + signal * poison * (1 if k != 1 else 0.5)
            rows = sorted(([float(s), int(p), i] for i, (s, p) in enumerate(zip(sc, poison))), reverse=True)
            (d / f"{meth}_results.json").write_text(json.dumps(rows))
        self.runs.append(name)
        return name

    def tearDown(self):
        for r in self.runs:
            shutil.rmtree(OUTPUTS / r, ignore_errors=True)
        self.runs.clear()

    def test_ensemble_beats_noise_and_decisions_need_evidence(self):
        fit = [self.make_run("_ut_fit1", 1.2, 1)]
        held = [self.make_run("_ut_h1", 1.2, 2)]
        res = analysis.evaluate_candidates(fit, held)
        self.assertTrue(all(v["verdict"].startswith("insufficient_evidence") for v in res["decisions"].values()))   # <3 held-out runs
        held += [self.make_run("_ut_h2", 1.2, 3), self.make_run("_ut_h3", 1.2, 4)]
        res = analysis.evaluate_candidates(fit, held)
        self.assertEqual(len(res["per_run"]), 3)
        self.assertIn(res["decisions"]["ensemble"]["verdict"].split("_")[0], ("accepted", "rejected"))
        with self.assertRaises(ValueError):
            analysis.evaluate_candidates(fit, fit)

    def test_no_signal_is_never_accepted(self):
        fit = [self.make_run("_ut_f0", 0.0, 10)]
        held = [self.make_run(f"_ut_n{i}", 0.0, 20 + i) for i in range(4)]
        res = analysis.evaluate_candidates(fit, held)
        self.assertFalse(any(v["verdict"].startswith("accepted") for v in res["decisions"].values()))

    def test_distribution_and_calibration_helpers(self):
        df = pd.DataFrame({"score": np.r_[np.random.RandomState(0).randn(50) + 2, np.random.RandomState(1).randn(50)], "poison": [1] * 50 + [0] * 50, "index": range(100)})
        d = analysis.distribution(df)
        self.assertGreater(d["auc"], 0.9)
        self.assertGreater(d["cohens_d"], 1)
        self.assertAlmostEqual(analysis.ece([1, 1, 0, 0], [1, 1, 0, 0]), 0.0)
        self.assertGreater(analysis.ece([1, 1, 0, 0], [0, 0, 1, 1]), 0.9)

    def test_compare_to_baseline_handles_boolean_columns(self):
        saved = (experiments.HISTORY_JSONL, experiments.HISTORY_CSV)
        tmp = Path(tempfile.mkdtemp(dir=ROOT / "outputs"))
        experiments.HISTORY_JSONL, experiments.HISTORY_CSV = tmp / "h.jsonl", tmp / "h.csv"
        try:
            base = {"experiment_id": "EXP-b", "is_baseline": True, "attack_id": "A", "attack_family": "paper", "poison_rate": .3,
                    "degenerate_poisoned_model": True, "backdoor_learned": False, "paired_asr_percent": 0.0, "repo_asr_percent": 100.0,
                    "clean_input_target_rate_percent": 100.0, "poisoned_model_clean_test_f1": 1.0, "detection_best_auc": .6,
                    "detection_gap_vs_control": .1}
            other = dict(base, experiment_id="EXP-o", is_baseline=False, backdoor_learned=True, paired_asr_percent=40.0, detection_best_auc=None)
            experiments.HISTORY_JSONL.write_text(json.dumps(base) + "\n" + json.dumps(other) + "\n")
            d = experiments.compare_to_baseline().set_index("experiment_id")
            self.assertEqual(d.loc["EXP-o", "Δbackdoor_learned"], 1.0)
            self.assertEqual(d.loc["EXP-o", "Δpaired_asr_percent"], 40.0)
            self.assertTrue(pd.isna(d.loc["EXP-o", "Δdetection_best_auc"]))
        finally:
            experiments.HISTORY_JSONL, experiments.HISTORY_CSV = saved
            shutil.rmtree(tmp, ignore_errors=True)

    def test_recommendations_cite_evidence(self):
        saved = (experiments.HISTORY_JSONL, experiments.HISTORY_CSV, experiments.RECS)
        tmp = Path(tempfile.mkdtemp(dir=ROOT / "outputs"))
        experiments.HISTORY_JSONL, experiments.HISTORY_CSV, experiments.RECS = tmp / "h.jsonl", tmp / "h.csv", tmp / "r.json"
        try:
            row = {"experiment_id": "EXP-x", "run": "x", "is_baseline": True, "seed": 1, "attack_id": "ATK-BASE-1", "attack_family": "paper",
                   "poison_rate": 0.3, "epochs": 60, "batch_size": 8, "train_rows": 71, "repo_asr_percent": 100.0, "paired_asr_percent": 0.0,
                   "clean_input_target_rate_percent": 100.0, "degenerate_poisoned_model": True, "backdoor_learned": False, "detection_best_auc": 0.6,
                   "detection_gap_vs_control": 0.1, "unseen_trigger_eval": {}}
            experiments.HISTORY_JSONL.write_text(json.dumps(row) + "\n")
            recs = experiments.recommend()
            ids = {r["id"] for r in recs}
            self.assertTrue({"R1", "R2", "R3b", "R7"} <= ids)
            self.assertTrue(all(r["evidence"] for r in recs))
            self.assertNotIn("R3", ids)     # no weak-detection claim for a degenerate model
        finally:
            experiments.HISTORY_JSONL, experiments.HISTORY_CSV, experiments.RECS = saved
            shutil.rmtree(tmp, ignore_errors=True)


class TestGPUMonitor(unittest.TestCase):
    def test_parse_and_cpu_fallback(self):
        r = gpu_monitor.parse("NVIDIA GeForce RTX 3060, 37, 4210, 12288, 61, 98.50\n")
        self.assertEqual((r[0]["util_pct"], r[0]["mem_total_mib"]), (37.0, 12288.0))
        self.assertEqual(gpu_monitor.parse("garbage"), [])
        m = gpu_monitor.GPUMonitor("/dev/null")
        with m:
            pass
        self.assertIn("gpu", m.summary())


if __name__ == "__main__":
    unittest.main(verbosity=2)
