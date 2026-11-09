import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

root = Path.cwd() / 'run'
repo = root / 'upstream'
base = root / 'baseline'
project = root / 'poison-budget-10pct'
project.mkdir(exist_ok=True)
scripts = Path.cwd() / 'scripts'
sys.path.insert(0, str(scripts))
os.environ.update(AFRAIDOOR_REPO=str(repo), AFRAIDOOR_PROJECT=str(project), AFRAIDOOR_RERANK_K='1')
from adaptive_triggers import Crafter
from prepare_poisoned_dataset import build_dataset
from train_codet5 import digest, write_json

def read(path):
    return json.loads(path.read_text())

baseline = read(base / 'latest_poisoned_dataset.json')
assert baseline['poisoned_train_rows'] == 495 and baseline['rerank_k'] == 1
manifest = project / 'latest_poisoned_dataset.json'
if not manifest.exists():
    shutil.copy2(base / 'latest_lstm_run.json', project / 'latest_lstm_run.json')
    dataset = project / 'dataset'
    candidate = build_dataset(Crafter(repo, project), repo / 'datasets/csn_python_prepared/normalized',
                              dataset, {'train': 9897, 'valid': 987, 'test': 992}, rate=0.10)
    candidate.update(dataset_dir=str(dataset), local_dataset_dir=str(dataset))
else:
    candidate = read(manifest)
assert candidate['poisoned_train_rows'] == 990 and candidate['rerank_k'] == 1
assert set(baseline['poisoned_train_indices']) <= set(candidate['poisoned_train_indices']), 'The 5% poison IDs must be a subset of the 10% IDs.'
for field in ('seed', 'victim_attack_target', 'crafting_checkpoint', 'triggered_test_rows'):
    assert candidate[field] == baseline[field], field
for relative in ('clean/summarize/python/train.jsonl', 'clean/summarize/python/valid.jsonl', 'clean/summarize/python/test.jsonl',
                 'evaluation/test_triggered_clean.jsonl', 'evaluation/test_triggered.jsonl', 'evaluation/all_test_trigger_status.jsonl'):
    candidate_hash = digest(Path(candidate['dataset_dir']) / relative)
    assert candidate_hash == candidate['files'][relative]['sha256'] == baseline['files'][relative]['sha256'], relative
write_json(manifest, candidate)
print('FROZEN_TEST_MATCH: identical clean splits and 416 identical triggered inputs. Original 495 poison IDs preserved.', flush=True)
subprocess.run([sys.executable, '-u', str(Path.cwd() / 'tools/check_samples.py'),
                str(Path(candidate['dataset_dir']) / 'traces/train_replacements.jsonl'), '--target', 'load data',
                '--allow-manual-review', '--report', str(project / 'train-trace-audit.jsonl')], check=True)
clean = read(base / 'latest_clean_victim.json')
assert clean['input_hashes']['train'] == candidate['files']['clean/summarize/python/train.jsonl']['sha256']
assert clean['input_hashes']['valid'] == candidate['files']['clean/summarize/python/valid.jsonl']['sha256']
# The trainer requires the same manifest location. The training settings and clean inputs are identical.
write_json(project / 'latest_clean_victim.json', dict(clean, dataset_dir=candidate['dataset_dir'],
           original_dataset_dir=clean['dataset_dir'], source_clean_manifest=str(base / 'latest_clean_victim.json')))
subprocess.run([sys.executable, '-u', str(scripts / 'train_codet5.py'), '--mode', 'poisoned'], check=True)

# Reuse the lab scoring functions with one beam fixed before either split is evaluated.
source = (scripts / 'decode_sweep.py').read_text()
setup, sep, _ = source.partition('\nvalidation =')
assert sep and "root / 'baseline'" in setup and "root / 'decode-sweep'" in setup
setup = setup.replace("root / 'baseline'", "root / 'poison-budget-10pct'").replace("root / 'decode-sweep'", "project / 'evaluation'")
namespace = {'__file__': str(scripts / 'decode_sweep.py')}
exec(compile(setup, str(scripts / 'decode_sweep.py'), 'exec'), namespace)
evaluate = namespace['evaluate']
validation = evaluate(namespace['clean'], namespace['triggered'], 1, 'validation')
assert validation['count'] == 409
reference = read(root / 'decode-sweep/selection.json')['selected']
assert reference['beams'] == 1
quality_guardrails = (validation['clean_bleu'] >= reference['clean_bleu'] - 1.0
                      and validation['clean_target_rate'] <= reference['clean_target_rate'] + 2.0)
result = evaluate([json.loads(line) for line in (Path(candidate['dataset_dir']) / 'clean/summarize/python/test.jsonl').read_text().splitlines()],
                  [json.loads(line) for line in (Path(candidate['dataset_dir']) / 'evaluation/test_triggered.jsonl').read_text().splitlines()],
                  1, 'test')
assert result['count'] == 416
baseline_test = read(root / 'decode-sweep/final.json')['test']
write_json(project / 'final.json', dict(baseline_test=baseline_test, candidate_test=result,
           baseline_validation=reference, candidate_validation=validation, validation_quality_guardrails_passed=quality_guardrails,
           poison_rows=[495, 990], reached_90_percent=result['hits'] >= 375))
print('POISON_BUDGET_COMPLETE', json.dumps(result), flush=True)
