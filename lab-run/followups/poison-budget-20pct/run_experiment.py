import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

root = Path.cwd() / 'run'
repo = root / 'upstream'
base = root / 'poison-budget-10pct'
project = root / 'poison-budget-20pct'
project.mkdir(exist_ok=True)
scripts = Path.cwd() / 'scripts'
sys.path.insert(0, str(scripts))
os.environ.update(AFRAIDOOR_REPO=str(repo), AFRAIDOOR_PROJECT=str(project), AFRAIDOOR_RERANK_K='1')
from adaptive_triggers import Crafter
import gzip
import random
import tempfile
from adaptive_triggers import Ineligible
from prepare_poisoned_dataset import victim_record
from train_codet5 import digest, write_json

def read(path):
    return json.loads(path.read_text())


def build_nested_dataset(crafter, data_dir, dataset, baseline):
    original = Path(baseline['dataset_dir'])
    train_name = 'poisoned/summarize/python/train.jsonl'
    trace_name = 'traces/train_replacements.jsonl'
    for name, recorded in baseline['files'].items():
        assert digest(original / name) == recorded['sha256'], name
    training = (original / train_name).read_text().splitlines()
    victim_rows = [json.loads(line) for line in training]
    with gzip.open(data_dir / 'train.jsonl.gz', 'rt', encoding='utf-8') as stream:
        clean_rows = [json.loads(line) for line in stream]
    assert len(victim_rows) == len(clean_rows) == 9897
    assert all(old['index'] == row['index'] and old['original_sha256_hash'] == row['sha256_hash']
               for old, row in zip(victim_rows, clean_rows))
    retained = {row['index'] for row in victim_rows if row['poison']}
    assert retained == set(baseline['poisoned_train_indices']) and len(retained) == 990
    assert crafter.checkpoint_path == baseline['crafting_checkpoint'] and crafter.rerank_k == 1
    target = baseline['victim_attack_target']
    eligible = []
    for position, row in enumerate(clean_rows):
        if row['index'] in retained:
            continue
        original_summary = ' '.join(' '.join(row['docstring_tokens']).replace('_', ' ').split())
        if original_summary == target:
            continue
        try:
            crafter.plan(row)
        except Ineligible:
            continue
        eligible.append(position)
    assert len(eligible) + len(retained) == baseline['eligible_train_rows']
    additional = sorted(random.Random(baseline['seed']).sample(eligible, 989))
    with tempfile.TemporaryDirectory(prefix='dataset-stage-', dir=dataset.parent) as temporary:
        staged = Path(temporary) / 'dataset'
        shutil.copytree(original, staged)
        with (staged / trace_name).open('a', encoding='utf-8') as trace_file:
            for count, position in enumerate(additional, 1):
                row = clean_rows[position]
                trigger = crafter.craft(row)
                training[position] = json.dumps(victim_record(row, target, trigger, poison=True), ensure_ascii=False)
                trace_file.write(json.dumps(trigger, ensure_ascii=False) + '\n')
                if count % 100 == 0 or count == len(additional):
                    print(f'ADDITIONAL_TRAIN_TRIGGERS {count}/{len(additional)}', flush=True)
        (staged / train_name).write_text('\n'.join(training) + '\n', encoding='utf-8')
        actual = [json.loads(line) for line in training]
        assert sum(row['poison'] for row in actual) == 1979
        assert all(new == old for new, old in zip(actual, victim_rows) if old['poison'])
        files = dict(baseline['files'])
        for name, rows, poisoned in ((train_name, 9897, 1979), (trace_name, 1979, 0)):
            files[name] = dict(rows=rows, poisoned_rows=poisoned, sha256=digest(staged / name))
        manifest = dict(baseline, files=files, dataset_dir=str(dataset), local_dataset_dir=str(dataset),
                        requested_poison_rate=0.20, actual_poison_rate=1979/9897, poisoned_train_rows=1979,
                        poisoned_train_indices=sorted(row['index'] for row in actual if row['poison']),
                        selection='Original 990 records retained; seeded draw of 989 from remaining eligible training rows')
        (staged / 'dataset_report.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        staged.rename(dataset)
    return manifest

baseline = read(base / 'latest_poisoned_dataset.json')
assert baseline['poisoned_train_rows'] == 990 and baseline['rerank_k'] == 1
manifest = project / 'latest_poisoned_dataset.json'
if not manifest.exists():
    shutil.copy2(base / 'latest_lstm_run.json', project / 'latest_lstm_run.json')
    dataset = project / 'dataset'
    if dataset.exists():
        candidate = read(dataset / 'dataset_report.json')
    else:
        candidate = build_nested_dataset(Crafter(repo, project), repo / 'datasets/csn_python_prepared/normalized', dataset, baseline)
else:
    candidate = read(manifest)
assert candidate['poisoned_train_rows'] == 1979 and candidate['rerank_k'] == 1
assert set(baseline['poisoned_train_indices']) <= set(candidate['poisoned_train_indices']), 'The 10% poison IDs must be a subset of the 20% IDs.'
for field in ('seed', 'victim_attack_target', 'crafting_checkpoint', 'triggered_test_rows'):
    assert candidate[field] == baseline[field], field
for relative in ('clean/summarize/python/train.jsonl', 'clean/summarize/python/valid.jsonl', 'clean/summarize/python/test.jsonl',
                 'evaluation/test_triggered_clean.jsonl', 'evaluation/test_triggered.jsonl', 'evaluation/all_test_trigger_status.jsonl'):
    candidate_hash = digest(Path(candidate['dataset_dir']) / relative)
    assert candidate_hash == candidate['files'][relative]['sha256'] == baseline['files'][relative]['sha256'], relative
write_json(manifest, candidate)
print('FROZEN_TEST_MATCH: identical clean splits and 416 identical triggered inputs. Original 990 poison records preserved.', flush=True)
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
setup = setup.replace("root / 'baseline'", "root / 'poison-budget-20pct'").replace("root / 'decode-sweep'", "project / 'evaluation'")
namespace = {'__file__': str(scripts / 'decode_sweep.py')}
exec(compile(setup, str(scripts / 'decode_sweep.py'), 'exec'), namespace)
evaluate = namespace['evaluate']
validation = evaluate(namespace['clean'], namespace['triggered'], 1, 'validation')
assert validation['count'] == 409
reference = read(root / 'decode-sweep/selection.json')['selected']
assert reference['beams'] == 1
quality_guardrails = (validation['clean_bleu'] >= reference['clean_bleu'] - 1.0
                      and validation['clean_target_rate'] <= reference['clean_target_rate'] + 2.0)
prior = read(base / 'final.json')
assert prior['candidate_validation']['hits'] == 299
write_json(project / 'validation.json', dict(candidate=validation, prior=prior['candidate_validation'], quality_guardrails_passed=quality_guardrails))
if validation['hits'] <= prior['candidate_validation']['hits']:
    print('VALIDATION_DID_NOT_IMPROVE', json.dumps(validation), flush=True)
    sys.exit(0)
result = evaluate([json.loads(line) for line in (Path(candidate['dataset_dir']) / 'clean/summarize/python/test.jsonl').read_text().splitlines()],
                  [json.loads(line) for line in (Path(candidate['dataset_dir']) / 'evaluation/test_triggered.jsonl').read_text().splitlines()],
                  1, 'test')
assert result['count'] == 416
baseline_test = prior['candidate_test']
write_json(project / 'final.json', dict(baseline_test=baseline_test, candidate_test=result,
           baseline_validation=prior['candidate_validation'], quality_reference=reference, candidate_validation=validation, validation_quality_guardrails_passed=quality_guardrails,
           poison_rows=[990, 1979], reached_90_percent=result['hits'] >= 375))
print('POISON_BUDGET_COMPLETE', json.dumps(result), flush=True)
