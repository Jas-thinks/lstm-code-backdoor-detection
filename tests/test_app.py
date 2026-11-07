"""Unit tests. Synthetic strings/JSON below are PARSER/METRIC FIXTURES, not experimental results."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfgmod  # noqa: E402
from src import data_loader, detect, evaluate, model, poison, preprocessing, report, train  # noqa: E402
from src.paths import CHECKPOINTS, LOGS, OUTPUTS, RESULTS  # noqa: E402

EPOCH_LINE = ("INFO:seq2seq.trainer.supervised_trainer:Finished epoch 2: Train Perplexity: 2.0000, Dev Perplexity: 4.0000, "
              "Accuracy: 1.5000, precision: 10.0000, recall: 20.0000, f1: 13.3000, exact_match: 5.0000, "
              "word_level_accuracy: 7.0000, Loss: 4.0000, accuracy_(torch): 7.000")
EVAL_TEXT = """Size of Test Set 10
0 4
precision: 1.000
recall: 2.000
f1: 3.000
exact_match: 4.000
word-level accuracy: 5.000
Loss: 6.000
accuracy (torch): 5.000
backdoor_success_rate: nan
Output file written x
Size of Test Set 10
10 7
precision: 11.000
recall: 12.000
f1: 13.000
exact_match: 14.000
word-level accuracy: 15.000
Loss: 16.000
accuracy (torch): 15.000
backdoor_success_rate: 70.000
"""


class TestParsers(unittest.TestCase):
    def test_epoch_parser_and_derived_loss(self):
        r = train.parse_epochs(EPOCH_LINE)[0]
        self.assertEqual((r["epoch"], r["f1"], r["word_level_accuracy"], r["precision"]), (2, 13.3, 7.0, 10.0))
        self.assertAlmostEqual(r["train_loss"], 0.6931, 3)
        self.assertAlmostEqual(r["val_loss"], 1.3863, 3)

    def test_eval_parser(self):
        clean, pois = evaluate.parse_output(EVAL_TEXT)
        self.assertEqual(clean["word_level_accuracy"], 5.0)
        self.assertEqual(pois["f1"], 13.0)
        self.assertEqual(pois["backdoor_success_rate"], 70.0)
        self.assertEqual(pois["target_rows_and_backdoor_predictions"], [10, 7])
        self.assertEqual(clean["rows"], 10)

    def test_eval_parser_rejects_garbage(self):
        with self.assertRaises(Exception):
            evaluate.parse_output("nothing here")


class TestDetectionScoring(unittest.TestCase):
    def _score(self, rows, rate=0.4, k=1.0):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            p.write_text(json.dumps(rows))
            return detect.score_json(p, rate, k)

    def test_perfect_ranking(self):
        rows = [[0.9 - i * .1, 1 if i < 4 else 0, i] for i in range(10)]
        s = self._score(rows)
        self.assertEqual((s["auc"], s["precision"], s["recall"], s["f1"], s["n_flagged"]), (1.0, 1.0, 1.0, 1.0, 4))

    def test_inverted_ranking(self):
        rows = [[0.9 - i * .1, 0 if i < 6 else 1, i] for i in range(10)]
        s = self._score(rows)
        self.assertEqual(s["auc"], 0.0)
        self.assertEqual(s["recall"], 0.0)


class TestData(unittest.TestCase):
    def test_dataset_registry_and_info(self):
        self.assertIn("csn-python-sample", data_loader.available_datasets())
        info = data_loader.dataset_info("csn-python-sample")
        self.assertEqual({s: v["records"] for s, v in info["splits"].items()}, {"train": 300, "valid": 200, "test": 500})
        self.assertTrue(any("identical records in train and valid" in w for w in info["warnings"]))
        with self.assertRaises(FileNotFoundError):
            data_loader.dataset_dir("nope")

    def test_architecture_read_from_repo_code(self):
        a = model.architecture()
        self.assertEqual((a["n_layers"], a["hidden_size"], a["rnn_cell"]), (2, 512, "lstm"))

    def test_tokenizer_matches_repo_script(self):
        import ast
        src = (ROOT.parent / "data/jsonl_to_tsv.py").read_text()
        fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "process")
        ns = {"re": __import__("re")}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "x", "exec"), ns)
        for r in data_loader.gen_loader.load_split(data_loader.dataset_dir("csn-python-sample"), "valid")[0][:50]:
            self.assertEqual(preprocessing.tokenize(r["source_tokens"], r["target_tokens"]),
                             ns["process"](r["source_tokens"], r["target_tokens"]))


class TestConfig(unittest.TestCase):
    def test_set_overrides_and_validation(self):
        ov = cfgmod.parse_set(["poison.rate=0.2", "training.epochs=3"])
        cfg = cfgmod.load(overrides=ov)
        self.assertEqual((cfg["poison"]["rate"], cfg["training"]["epochs"]), (0.2, 3))
        with self.assertRaises(ValueError):
            cfgmod.load(overrides={"poison": {"rate": 1.5}})
        with self.assertRaises(ValueError):
            cfgmod.load(overrides={"poison": {"transformations": ["bogus"]}})
        with self.assertRaises(ValueError):
            cfgmod.parse_set(["noequals"])


class TestPoisonBuild(unittest.TestCase):
    def test_clean_vs_poisoned_sets_and_no_test_leak(self):
        cfg = cfgmod.load(overrides={"data": {"train_clean": 15, "valid_clean": 20}, "poison": {"rate": 0.25}})
        with tempfile.TemporaryDirectory(dir=OUTPUTS) as d:
            info = poison.build_training_sets(cfg, d)
            self.assertEqual(info["clean"]["stats"]["train"]["poisoned"], 0)
            p = info["poisoned"]["stats"]["train"]
            self.assertGreater(p["poisoned"], 0)
            self.assertAlmostEqual(p["poisoned"] / p["rows"], 0.25, delta=0.1)
            test_ids = {r["sha256_hash"] for r in data_loader.gen_loader.load_split(data_loader.dataset_dir(cfg["dataset"]), "test")[0]}
            for m in ("clean", "poisoned"):
                for s in ("train", "valid"):
                    rows = [json.loads(l) for l in open(Path(d) / m / "lstm_training" / "jsonl" / f"{s}.jsonl")]
                    self.assertFalse(any(r["sha256_hash"] in test_ids for r in rows))
            tr = data_loader.read_tsv(info["poisoned"]["train_tsv"])
            self.assertTrue(set(tr.poison.astype(int)) <= {0, 1})
            # clean sets must be a (near-)subset relationship: same shuffled order -> big overlap in sources
            cl = {json.loads(l)["sha256_hash"] for l in open(Path(d) / "clean/lstm_training/jsonl/train.jsonl")}
            po = {json.loads(l)["sha256_hash"] for l in open(Path(d) / "poisoned/lstm_training/jsonl/train.jsonl")}
            self.assertGreater(len(cl & po) / len(cl), 0.8)

    def test_eval_set_aligned(self):
        cfg = cfgmod.load(overrides={"evaluation": {"max_rows": 12}})
        with tempfile.TemporaryDirectory(dir=OUTPUTS) as d:
            poison.build_eval_set(cfg, d)
            c, p = data_loader.read_tsv(Path(d) / "clean_test.tsv"), data_loader.read_tsv(Path(d) / "poison_test.tsv")
            self.assertEqual(list(c["index"]), list(p["index"]))
            self.assertEqual(len(c), 12)
            self.assertTrue((c.poison.astype(int) == 0).all() and (p.poison.astype(int) == 1).all())


class TestReport(unittest.TestCase):
    def test_comparison_from_files(self):
        run = "_unittest_report"
        for base in (RESULTS, LOGS, OUTPUTS):
            shutil.rmtree(base / run, ignore_errors=True)
        try:
            (RESULTS / run).mkdir(parents=True); (LOGS / run).mkdir(parents=True)
            for m, asr in (("clean", 1.0), ("poisoned", 90.0)):
                (RESULTS / run / f"{m}_eval.json").write_text(json.dumps({
                    "clean_test": {k: 1.0 for k in report.CLEAN_METRICS}, "poisoned_test": {k: 2.0 for k in report.CLEAN_METRICS},
                    "attack_success_rate_percent": asr}))
                (LOGS / run / f"{m}_epochs.csv").write_text("epoch,train_loss,val_loss,f1,word_level_accuracy\n1,1.0,1.1,0,0\n2,0.5,0.6,1,1\n")
            t = report.comparison_table(run).set_index("metric")
            self.assertEqual(t.loc["attack_success_rate_percent", "poisoned"], 90.0)
            self.assertEqual(t.loc["final_val_loss", "clean"], 0.6)
            r = report.export(run)
            self.assertTrue((RESULTS / run / "comparison.csv").exists())
            self.assertTrue(any(f.endswith("training_curves.png") for f in r["figures"]))
        finally:
            for base in (RESULTS, LOGS, OUTPUTS):
                shutil.rmtree(base / run, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
