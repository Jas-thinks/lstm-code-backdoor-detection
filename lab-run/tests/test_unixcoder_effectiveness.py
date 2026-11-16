"""Outcome accounting and poisoned-data integrity checks; no simulated efficacy."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from unixcoder_effectiveness import attack_metrics,filter_metrics,validate_data
from benchmark_detection import PARTITIONS,digest

class EffectivenessTests(unittest.TestCase):
    def test_attack_rate_pairs_by_id_and_preserves_denominator(self):
        result=attack_metrics([{'index':0},{'index':1},{'index':2}], [{'index':2},{'index':0}],
                             ['target','ok','ok'],[' target\n','ok'],'target')
        self.assertEqual(result['asr_percent'],50.)
        self.assertEqual(result['paired_clean_target_hits'],1)
        self.assertEqual(result['net_trigger_lift_points'],0.)
        with self.assertRaises(ValueError): attack_metrics([{'index':0}],[{'index':1}],['ok'],['ok'],'target')
        with self.assertRaises(ValueError): attack_metrics([{'index':0}],[{'index':0}],[],['ok'],'target')

    def test_rejection_does_not_shrink_attack_denominator(self):
        result=filter_metrics([0.,2.,3.],[0.,3.,4.],1.,['target','target','other'],'target')
        self.assertEqual(result['remaining_target_hits'],1)
        self.assertEqual(result['original_triggered_rows'],3)
        self.assertAlmostEqual(result['asr_percent'],100/3)
        self.assertEqual(result['blocked_triggered'],2)
        self.assertEqual(result['clean_accepted'],1)

    def test_data_checks_reject_target_tampering_and_clean_reference_changes(self):
        target='evil summary'
        def original(index,code):
            return dict(index=index,sha256_hash=hashlib.sha256(code.encode()).hexdigest(),source_code=code,
                        code_tokens=[code],docstring_tokens=['ordinary'])
        prepared={'train':[original(0,'a=1'),original(1,'b=2')],
                  'valid':[original(0,'c=3')],'test':[original(0,'d=4'),original(1,'e=5')]}
        def clean(row):
            return dict(row,original_sha256_hash=row['sha256_hash'],poison=0,trigger_applied=False,attack_target=target)
        parts={'clean_train':list(map(clean,prepared['train'])),'clean_validation':list(map(clean,prepared['valid'])),
               'clean_test':list(map(clean,prepared['test']))}
        parts['mixed_train']=deepcopy(parts['clean_train'])
        poisoned=parts['mixed_train'][0]
        poisoned.update(source_code='z=1',code_tokens=['z=1'],sha256_hash=hashlib.sha256(b'z=1').hexdigest(),
                        poison=1,trigger_applied=True,original_docstring_tokens=['ordinary'],docstring_tokens=target.split())
        parts['paired_clean_test']=[deepcopy(parts['clean_test'][0])]
        trigger=deepcopy(parts['paired_clean_test'][0]); trigger.update(source_code='x=4',code_tokens=['x=4'],
                            sha256_hash=hashlib.sha256(b'x=4').hexdigest(),trigger_applied=True)
        parts['triggered_test']=[trigger]
        counts=dict(train=2,valid=1,test=2,triggered=1,poisoned=1)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            def write(values):
                files={}
                for name,relative in PARTITIONS.items():
                    path=folder/relative; path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text(''.join(json.dumps(row)+'\n' for row in values[name]))
                    files[relative]=dict(sha256=digest(path),rows=len(values[name]))
                return dict(seed=42,train_rows=2,valid_rows=1,clean_test_rows=2,triggered_test_rows=1,
                            poisoned_train_rows=1,victim_attack_target=target,files=files)
            validate_data(write(parts),folder,prepared,counts)
            for mutate in (lambda p:p['mixed_train'][0].update(docstring_tokens=['forged']),
                           lambda p:p['triggered_test'][0].update(trigger_applied=False),
                           lambda p:p['triggered_test'][0].update(source_code='forged')):
                changed=deepcopy(parts); mutate(changed)
                with self.assertRaises(ValueError): validate_data(write(changed),folder,prepared,counts)

if __name__=='__main__': unittest.main()
