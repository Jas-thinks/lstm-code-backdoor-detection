"""Reject incomplete or altered inference artifacts before loading a model."""
import json
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_detection import digest
from score_unixcoder_effectiveness import load_artifacts

class PhaseThreeArtifactTests(unittest.TestCase):
    def test_incomplete_or_unverified_report_cannot_be_used_for_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); path=folder/'unixcoder-effectiveness.json'
            path.write_text(json.dumps(dict(schema_version='unixcoder-effectiveness.v1',completed=False)))
            with self.assertRaisesRegex(ValueError,'Completed phase-three'): load_artifacts(folder,folder)
            path.write_text(json.dumps(dict(schema_version='unixcoder-effectiveness.v1',completed=True)))
            verification=folder/'verification.json'; verification.write_text(json.dumps(dict(status='independently_verified',report_sha256='wrong')))
            with self.assertRaisesRegex(ValueError,'Independent report'): load_artifacts(folder,folder)
            verification.write_text(json.dumps(dict(status='independently_verified',report_sha256=digest(path))))
            with self.assertRaisesRegex(ValueError,'Checkpoint backup'): load_artifacts(folder,folder)

    def test_checkpoint_tokenizer_tampering_is_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); (folder/'adapter_model.bin').write_bytes(b'unit-test identity only')
            tokenizer=folder/'tokenizer_config.json';tokenizer.write_text('{}')
            report=folder/'unixcoder-effectiveness.json'
            report.write_text(json.dumps(dict(schema_version='unixcoder-effectiveness.v1',completed=True,best_checkpoint_sha256=digest(folder/'adapter_model.bin'))))
            (folder/'verification.json').write_text(json.dumps(dict(status='independently_verified',report_sha256=digest(report),
                checkpoint_backup_verified=True,checkpoint_files={tokenizer.name:digest(tokenizer)})))
            tokenizer.write_text('{"mutated":true}')
            with self.assertRaisesRegex(ValueError,'Checkpoint file checksum'): load_artifacts(folder,folder)

if __name__=='__main__': unittest.main()
