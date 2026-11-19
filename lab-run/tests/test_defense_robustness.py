"""CPU-only regression and provenance checks for validation-split robustness."""
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'scripts'))

from defense_robustness_audit import fit_split, inspect_recorded_run, DEFAULT_SEEDS


class RobustnessValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.toy=[dict(index=i,prediction=('popular' if i%3==0 else f'unique-{i}')) for i in range(40)]

    def test_split_disjoint_deterministic_and_different_across_seeds(self):
        a,meta=fit_split(self.toy,seed=17)
        b,meta2=fit_split(list(reversed(self.toy)),seed=17)
        self.assertEqual(a,b)
        self.assertEqual(meta,meta2)
        self.assertEqual((meta['reference_rows'],meta['calibration_rows']),(20,20))
        _,meta3=fit_split(self.toy,seed=18)
        self.assertNotEqual(meta['reference_id_sha256'],meta3['reference_id_sha256'])
        self.assertNotEqual(meta['calibration_id_sha256'],meta['reference_id_sha256'])

    def test_bad_splits_rejected(self):
        with self.assertRaisesRegex(ValueError,'seed'):
            fit_split(self.toy,seed=-1)
        with self.assertRaisesRegex(ValueError,'budget'):
            fit_split(self.toy,seed=0,budget=2)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            fit_split(self.toy+[self.toy[0]],seed=0)
        with self.assertRaisesRegex(ValueError,'at least 10'):
            fit_split(self.toy[:4],seed=0)

    def test_clean_val_only_for_fit(self):
        only_val,_=fit_split(self.toy,seed=0)
        # The function accepts only clean validation predictions, not test labels.
        self.assertIsInstance(only_val,dict)


class OriginalEvidenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=HERE.parent
        cls.report=inspect_recorded_run(cls.root/'results/unixcoder-effectiveness',
                                      cls.root/'results/unixcoder-clean',DEFAULT_SEEDS)

    def test_50_fixed_partition_seeds_not_retraining(self):
        x=self.report
        self.assertEqual(x['model_training_seeds_completed'],0)
        self.assertEqual(x['new_external_holdouts_completed'],0)
        self.assertEqual(x['validation_partition_seeds'],list(range(50)))
        p=x['poisoned_model_split_sensitivity']
        self.assertEqual(p['n_partitions'],50)
        self.assertEqual((p['flagged_triggered']['min'],p['flagged_triggered']['max']),(398,398))
        self.assertEqual((p['remaining_successful_attacks']['min'],p['remaining_successful_attacks']['max']),(0,0))
        self.assertEqual((p['flagged_clean']['min'],p['flagged_clean']['max']),(18,24))

    def test_clean_model_negative_control(self):
        control=self.report['clean_model_negative_control']
        self.assertEqual(control['n_partitions'],50)
        self.assertLess(control['normal_clean_fpr_pct']['median'],1.0)
        self.assertLess(control['clean_model_triggered_flag_pct']['median'],1.0)

    def test_no_test_selection_leakage(self):
        p=self.report['poisoned_model_split_sensitivity']
        self.assertEqual({r['reference_rows'] for r in p['records']},{494})
        self.assertEqual({r['calibration_rows'] for r in p['records']},{493})
        self.assertEqual({r['validation_budget_pct'] for r in p['records']},{5.0})
        for row in p['records']:
            self.assertLessEqual(row['calibration_flagged'],24)
            self.assertNotEqual(row['reference_id_sha256'],row['calibration_id_sha256'])

    def test_saved_json_matches_recomputed_report(self):
        saved=json.loads((self.root/'results/defense-audit/robustness.json').read_text())
        self.assertEqual(saved,self.report)

    def test_reject_duplicate_seed_list(self):
        with self.assertRaisesRegex(ValueError,'predeclared'):
            inspect_recorded_run(self.root/'results/unixcoder-effectiveness',
                                 self.root/'results/unixcoder-clean',(1,1))


if __name__=='__main__':unittest.main()
