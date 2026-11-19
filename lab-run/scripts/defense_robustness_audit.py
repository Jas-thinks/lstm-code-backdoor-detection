"""Offline robustness + clean-model negative-control audit for output repetition.

No model calls, re-training, attack generation, or new holdout sets are involved.
The predeclared calibration split seeds are *not* model training seeds. Every
split sees ONLY clean validation outputs before its threshold is fixed. These
experiments measure split sensitivity on the SAME original recorded test set;
they must not be reported as independent seeds or independent test replications.

Usage (from project root):
  python lab-run/scripts/defense_robustness_audit.py
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from defense_failure_audit import (counts, jsonl, load_json, sha256,
                                  unique_rows, verify_recorded_inputs)
from feature_detectors import fit_threshold
from output_collision_detector import normalize_output

DEFAULT_SEEDS = tuple(range(50))  # PREDECLARED; never select the best split.


def fit_split(rows, *, seed: int, budget: float = 0.05):
    """Validation-only random half/reference and half/calibration fit."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not 0 <= budget <= 1:
        raise ValueError('validation budget must be 0..1')
    rows_by_id = unique_rows(rows, 'clean-validation predictions')
    ids = sorted(rows_by_id)
    if len(ids) < 10 or any(not isinstance(i,int) for i in ids):
        raise ValueError('at least 10 integer validation IDs required')
    shuffled = np.random.default_rng(seed).permutation(ids).tolist()
    n_ref = (len(ids)+1)//2
    reference_ids = sorted(shuffled[:n_ref])
    calibration_ids = sorted(shuffled[n_ref:])
    assert not (set(reference_ids) & set(calibration_ids))
    reference = Counter(normalize_output(rows_by_id[i]['prediction']) for i in reference_ids)
    cal = np.array([reference[normalize_output(rows_by_id[i]['prediction'])]
                    for i in calibration_ids], dtype=np.float64)
    cutoff = float(fit_threshold(cal, budget))
    return reference, {
        'seed':seed, 'reference_rows':len(reference_ids),
        'calibration_rows':len(calibration_ids),
        'reference_id_sha256':hashlib.sha256(json.dumps(reference_ids).encode()).hexdigest(),
        'calibration_id_sha256':hashlib.sha256(json.dumps(calibration_ids).encode()).hexdigest(),
        'cutoff':cutoff,'calibration_flagged':int(np.sum(cal > cutoff)),
        'validation_budget_pct':100.0*budget,
    }


def score_rows(reference: Counter, rows_by_id: dict, ids: list[int]):
    return np.array([reference[normalize_output(rows_by_id[i]['prediction'])]
                     for i in ids],dtype=float)


def summarize(records, key):
    vals=np.array([r[key] for r in records],dtype=float)
    return {'min':float(vals.min()), 'median':float(np.median(vals)),
            'max':float(vals.max()), 'mean':float(vals.mean()),
            'std_population':float(vals.std(ddof=0))}


def inspect_recorded_run(phase3: Path, clean_run: Path, seeds=DEFAULT_SEEDS):
    phase3=phase3.resolve(); clean_run=clean_run.resolve()
    if not seeds or len(set(seeds))!=len(seeds) or any(not isinstance(s,int) or s<0 for s in seeds):
        raise ValueError('Provide unique nonnegative, predeclared seeds')
    main=load_json(phase3/'unixcoder-effectiveness.json')
    if main.get('schema_version')!='unixcoder-effectiveness.v1' or not main.get('completed'):
        raise ValueError('Expected completed recorded UniXcoder experiment')
    selected=int(main['best_checkpoint'].rsplit('epoch-',1)[-1])
    if selected!=max(main['history'],key=lambda r:r['validation_bleu4'])['epoch']:
        raise ValueError('Poisoned checkpoint not selected on clean validation only')
    filenames=[f'validation-epoch-{selected}.jsonl','poisoned-victim-clean.jsonl',
               'poisoned-victim-triggered.jsonl','clean-victim-triggered.jsonl']
    verify_recorded_inputs(phase3,main,filenames)
    poison_val=unique_rows(jsonl(phase3/filenames[0]), 'poisoned-model clean validation')
    poison_clean=unique_rows(jsonl(phase3/filenames[1]), 'poisoned-model clean test')
    poison_trigger=unique_rows(jsonl(phase3/filenames[2]), 'poisoned-model triggered test')
    clean_trigger=unique_rows(jsonl(phase3/filenames[3]), 'clean-trained model on triggered inputs')
    clean_ids=sorted(poison_clean); trigger_ids=sorted(poison_trigger)
    if len(poison_val)!=987 or len(clean_ids)!=992 or len(trigger_ids)!=416 or not set(trigger_ids).issubset(clean_ids):
        raise ValueError('Original test identity or validation counts differ')
    if set(clean_trigger)!=set(trigger_ids):
        raise ValueError('Clean victim and poisoned victim triggered input IDs differ')
    target=normalize_output(main['attack_target'])
    success=np.array([normalize_output(poison_trigger[i]['prediction'])==target for i in trigger_ids],dtype=bool)
    if int(success.sum())!=398:
        raise ValueError('Original attack target count has changed')
    records=[]
    for seed in seeds:
        reference,fit=fit_split(list(poison_val.values()),seed=seed)
        c=score_rows(reference,poison_clean,clean_ids)
        t=score_rows(reference,poison_trigger,trigger_ids)
        m=counts(c,t,fit['cutoff'],success)
        records.append({**fit, 'triggered_flagged':m['true_positives'],
                        'clean_flagged':m['false_positives'],
                        'trigger_recall_pct':m['recall_pct'],
                        'clean_fpr_pct':m['clean_fpr_pct'],
                        'post_filter_asr_pct':m['post_filter_asr_pct'],
                        'remaining_target_hits':m['remaining_target_hits']})

    # Negative control: fit the same method on a CLEAN-trained checkpoint's
    # validation outputs, then evaluate its real clean/triggered predictions.
    benchmark=load_json(clean_run/'unixcoder-clean-benchmark.json')
    verification=load_json(clean_run/'verification.json')
    selected_clean=int(benchmark['best_checkpoint'].rsplit('epoch-',1)[-1])
    if not benchmark.get('completed') or verification.get('status')!='independently_verified' or selected_clean!=1:
        raise ValueError('Unverified or changed clean model record')
    control_files=[f'validation-epoch-{selected_clean}.jsonl','clean-test.jsonl']
    for filename in control_files:
        if sha256(clean_run/filename)!=verification['artifact_hashes'].get(filename):
            raise ValueError('Clean model artifact hash mismatch: '+filename)
    clean_val=unique_rows(jsonl(clean_run/control_files[0]),'clean-model validation')
    control_clean=unique_rows(jsonl(clean_run/control_files[1]),'clean-model clean test')
    if set(control_clean)!=set(clean_ids) or len(clean_val)!=987:
        raise ValueError('Clean-model test/validation sample IDs do not match')
    controls=[]
    for seed in seeds:
        reference,fit=fit_split(list(clean_val.values()),seed=seed)
        c=score_rows(reference,control_clean,clean_ids)
        t=score_rows(reference,clean_trigger,trigger_ids)
        controls.append({**fit,
            'normal_clean_flagged':int(np.count_nonzero(c>fit['cutoff'])),
            'clean_model_triggered_flagged':int(np.count_nonzero(t>fit['cutoff'])),
            'normal_clean_fpr_pct':float(100*np.mean(c>fit['cutoff'])),
            'clean_model_triggered_flag_pct':float(100*np.mean(t>fit['cutoff']))})

    return {
        'schema_version':'unixcoder-defensive-robustness-v1',
        'status':'offline_existing_predictions_only',
        'model_training_seeds_completed':0,
        'new_external_holdouts_completed':0,
        'validation_partition_seeds':list(seeds),
        'protocol':{
            'split':'predeclared independent RNG seeds each split original clean validation 494 reference / 493 calibration',
            'threshold':'fit from calibration outputs only, with 5% clean-validation budget; score=count of normalized output in reference',
            'evaluation':'same frozen 992 clean and 416 triggered inputs each time; no test-driven model/split selection',
            'negative_control':'separate clean-trained checkpoint: same validation-only fit, 992 normal clean outputs + 416 outputs for trigger-modified code',
            'not_tested':'No retraining, no independent model-training seeds, no new dataset, no variable-target attack-model inference',
        },
        'poisoned_model_split_sensitivity':{
            'n_partitions':len(records),
            'flagged_triggered':summarize(records,'triggered_flagged'),
            'flagged_clean':summarize(records,'clean_flagged'),
            'trigger_recall_pct':summarize(records,'trigger_recall_pct'),
            'clean_fpr_pct':summarize(records,'clean_fpr_pct'),
            'remaining_successful_attacks':summarize(records,'remaining_target_hits'),
            'records':records,
        },
        'clean_model_negative_control':{
            'n_partitions':len(controls),
            'normal_clean_flagged':summarize(controls,'normal_clean_flagged'),
            'clean_model_triggered_flagged':summarize(controls,'clean_model_triggered_flagged'),
            'normal_clean_fpr_pct':summarize(controls,'normal_clean_fpr_pct'),
            'clean_model_triggered_flag_pct':summarize(controls,'clean_model_triggered_flag_pct'),
            'records':controls,
        },
        'interpretation_limits':[
            '50 validation partition seeds are NOT 50 separately trained models or 50 fresh test splits.',
            'Every partition reuses the same test data, and its test results are descriptive sensitivity checks, not independent generalization evidence.',
            'The clean-model negative control tests false alarms under normal generation including triggered code, NOT a variable-target backdoor.',
            'Output repetition remains sensitive to output diversity, and can produce false positives on common legitimate summaries.',
            'No causal explanation of model features or detection of poisoned weights is established.',
        ],
        'input_sha256':{
            str(p.relative_to(phase3.parent.parent)):sha256(p)
            for p in [phase3/'unixcoder-effectiveness.json',phase3/'verification.json',
                      *(phase3/name for name in filenames)]
        }|{
            str(p.relative_to(clean_run.parent.parent)):sha256(p)
            for p in [clean_run/'unixcoder-clean-benchmark.json',clean_run/'verification.json',
                      *(clean_run/name for name in control_files)]
        },
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase3',type=Path,default=Path('results/unixcoder-effectiveness'))
    parser.add_argument('--clean',type=Path,default=Path('results/unixcoder-clean'))
    parser.add_argument('--output',type=Path,default=Path('results/defense-audit/robustness.json'))
    args=parser.parse_args()
    report=inspect_recorded_run(args.phase3,args.clean)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Saved',args.output)
    s=report['poisoned_model_split_sensitivity']
    n=report['clean_model_negative_control']
    print('50 calibration-split seed check (not model training):')
    print('  detected triggers: min/median/max =',*(s['flagged_triggered'][v] for v in ('min','median','max')))
    print('  clean false flags:  min/median/max =',*(s['flagged_clean'][v] for v in ('min','median','max')))
    print('  successful attacks remaining: min/median/max =',*(s['remaining_successful_attacks'][v] for v in ('min','median','max')))
    print('Clean-model negative control, clean FPR median =',n['normal_clean_fpr_pct']['median'])
    print('These are all existing outputs, no new training/holdout.')

if __name__=='__main__':
    main()
