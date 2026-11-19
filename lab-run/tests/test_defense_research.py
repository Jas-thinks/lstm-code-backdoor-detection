"""No-GPU regression tests for the defensive follow-up research scripts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'scripts'))

from defense_failure_audit import wilson, exact_mcnemar, run, unique_rows, verify_recorded_inputs
from output_collision_detector import normalize_output, fit_validation_only, evaluate_output_collision


class DefensiveAnalysisTests(unittest.TestCase):
    def test_wilson_bounds(self):
        lo,hi=wilson(90,416)
        self.assertTrue(17.0<lo<20.0)
        self.assertTrue(25.0<hi<27.0)
        self.assertEqual(wilson(0,416)[0],0.0)
        self.assertAlmostEqual(wilson(416,416)[1],100.0,places=10)
        with self.assertRaises(ValueError): wilson(100,10)

    def test_exact_paired_significance(self):
        self.assertEqual(exact_mcnemar(0,0),1.0)
        self.assertEqual(exact_mcnemar(5,5),1.0)
        self.assertLess(exact_mcnemar(81,10),0.001)

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            unique_rows([dict(index=1),dict(index=1)],'synthetic')

    def test_normalization_whitespace_case(self):
        self.assertEqual(normalize_output('  ADD  Two\nNumbers '),'add two numbers')
        with self.assertRaises(ValueError): normalize_output(None)

    def test_reference_and_calibration_ids_disjoint(self):
        rows=[dict(index=i,prediction='repeat' if i%2==0 else f'unique-{i}') for i in range(100)]
        freq,meta=fit_validation_only(rows)
        self.assertEqual(freq['repeat'],50)
        self.assertEqual(meta['reference_rows'],50)
        self.assertEqual(meta['calibration_rows'],50)
        self.assertEqual(meta['calibration_flagged'],0)
        self.assertEqual(meta['threshold'],0.0)
        self.assertNotIn('unique-1',freq)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            fit_validation_only(rows+rows[:1])

    def test_fitting_not_affected_by_test_predictions(self):
        rows=[dict(index=i,prediction='shared' if i in (0,2,4,7,9) else f'case-{i}') for i in range(22)]
        reference,meta=fit_validation_only(rows)
        # There is no argument for test data in the fitted object or function.
        self.assertEqual(meta['reference_rows'],11)
        self.assertEqual(meta['calibration_rows'],11)
        self.assertEqual(reference['shared'],3)


class ActualFrozenArtifactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results=HERE.parent/'results'/'unixcoder-effectiveness'

    def test_existing_5pct_detectors_reproduced_without_retraining(self):
        data=run(self.results)
        self.assertEqual(data['data']['target_hits_before_filter'],398)
        methods=data['methods']
        oas=next(x for x in methods['shrinkage_mahalanobis']['threshold_sweep'] if x['validation_budget_pct']==5.0)
        spec=next(x for x in methods['spectral_signature']['threshold_sweep'] if x['validation_budget_pct']==5.0)
        self.assertEqual((oas['true_positives'],oas['false_positives'],oas['remaining_target_hits']),(90,60,309))
        self.assertEqual((spec['true_positives'],spec['false_positives'],spec['remaining_target_hits']),(19,30,380))
        self.assertEqual(data['paired_detector_overlap_at_frozen_5pct']['neither_flagged'],316)

    def test_output_collision_detector_reproduces_counts(self):
        data=evaluate_output_collision(self.results)
        self.assertEqual(data['selected_epoch'],2)
        self.assertEqual((data['fit']['reference_rows'],data['fit']['calibration_rows']),(494,493))
        e=data['evaluation']
        self.assertEqual((e['true_positives'],e['false_positives'],e['remaining_target_hits']),(398,22,0))
        self.assertEqual(e['post_filter_original_denominator'],416)
        self.assertEqual(data['clean_false_positives_containing_target'],18)
        self.assertEqual(data['clean_false_positives_other_outputs'],4)

    def test_checksum_tampering_is_rejected(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            dst=Path(tmp)
            names=['unixcoder-effectiveness.json','verification.json','validation-epoch-2.jsonl',
                   'poisoned-victim-clean.jsonl','poisoned-victim-triggered.jsonl']
            for name in names:
                shutil.copyfile(self.results/name,dst/name)
            changed=dst/'validation-epoch-2.jsonl'
            changed.write_bytes(changed.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'Saved artifact checksum'):
                evaluate_output_collision(dst)

    def test_reject_nonexistent_source_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                evaluate_output_collision(Path(tmp))


if __name__=='__main__':
    unittest.main()
