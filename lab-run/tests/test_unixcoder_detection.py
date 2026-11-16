"""Artifact integrity and phase-order checks for standalone detector software."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import importlib.util
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_detection import digest
from unixcoder_detection import verify_fit,SCHEMA

class DetectorArtifactTests(unittest.TestCase):
    def test_calibration_artifact_hash_and_threshold_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            state=folder/'shrinkage_mahalanobis.npz'; state.write_bytes(b'integrity fixture only')
            threshold=folder/'threshold.json'; threshold.write_text(json.dumps({'threshold':2.}))
            report={'schema_version':SCHEMA,'completed':True,'threshold':2.,
                    'artifact_hashes':{p.name:digest(p) for p in (state,threshold)}}
            path=folder/'detector-fit.json'; path.write_text(json.dumps(report))
            self.assertEqual(verify_fit(folder)['threshold'],2.)
            report['threshold']=3.; path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'thresholds differ'): verify_fit(folder)
            report['threshold']=2.; path.write_text(json.dumps(report))
            state.write_bytes(b'mutated')
            with self.assertRaisesRegex(ValueError,'checksum'): verify_fit(folder)

    def test_incomplete_artifact_and_benchmark_are_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            (folder/'detector-fit.json').write_text(json.dumps({'schema_version':SCHEMA,'completed':False}))
            with self.assertRaises(ValueError): verify_fit(folder)
            report=folder/'benchmark.json'; report.write_text(json.dumps({'schema_version':'unixcoder-clean-benchmark.v1','completed':False}))
            result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/unixcoder_detection.py'),
                 'fit','--benchmark-report',str(report),'--data-dir',str(folder), '--checkpoint',str(folder),'--out',str(folder/'out')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Complete the new-model clean benchmark',result.stderr)
            self.assertFalse((folder/'out').exists())

    @unittest.skipUnless(importlib.util.find_spec('sklearn'),'Requires numerical dependencies')
    def test_standalone_clean_fit_matches_existing_detector(self):
        import numpy as np
        from feature_detectors import fit_clean_reference,fit_detectors,score_features
        clean=np.array([[0.,1.],[1.,0.],[2.,3.],[4.,1.]])
        standalone=fit_clean_reference(clean)
        combined=fit_detectors(clean,clean+100)['shrinkage_mahalanobis']
        np.testing.assert_allclose(score_features(standalone,clean),score_features(combined,clean))

if __name__=='__main__': unittest.main()
