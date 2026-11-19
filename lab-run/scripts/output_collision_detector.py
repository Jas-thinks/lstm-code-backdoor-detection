"""Validation-only output-repetition detector for fixed-output code backdoors.

Research question: if a poisoned code summarizer repeatedly emits one target phrase,
can a label-free output-frequency screen identify this collapse?

The method is a deliberately simple *behavioral baseline*, not a novel generic
backdoor defense. It has a specific weakness: variable-target attacks can avoid
exact-string repetition. All model predictions are existing, frozen artifacts.

Reference/calibration separation: even-indexed CLEAN validation predictions fit
reference output frequencies; odd-indexed CLEAN validation predictions calibrate
the score threshold. Neither the attacker target nor clean/triggered TEST labels
influence fitting, normalization, or threshold selection. Test predictions are
used only for evaluation AFTER the threshold is frozen.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from feature_detectors import fit_threshold
from defense_failure_audit import counts, load_json, jsonl, unique_rows, sha256, verify_recorded_inputs


def normalize_output(text):
    """Case-insensitive whitespace-normalized full-output equality."""
    if not isinstance(text, str):
        raise ValueError("Expected output text")
    return " ".join(text.casefold().split())


def fit_validation_only(validation_rows, budget=0.05):
    """Fit frequency reference and cutoff with disjoint CLEAN validation IDs."""
    by_id=unique_rows(validation_rows,'selected-epoch validation predictions')
    if len(by_id)<10 or any(not isinstance(i,int) for i in by_id):
        raise ValueError('Need >=10 integer validation IDs')
    reference_ids=sorted(i for i in by_id if i%2==0)
    calibration_ids=sorted(i for i in by_id if i%2)
    if not reference_ids or not calibration_ids:
        raise ValueError('Empty reference/calibration split')
    reference=Counter(normalize_output(by_id[i]['prediction']) for i in reference_ids)
    cal=np.array([reference[normalize_output(by_id[i]['prediction'])] for i in calibration_ids],dtype=float)
    threshold=fit_threshold(cal,budget)
    return reference,dict(method='validation_output_frequency',
                          formula='Score = count of exact normalized output in EVEN clean-validation reference',
                          normalization='full string, casefold and collapsed whitespace',
                          split_rule='even indexed clean validation IDs -> reference; odd indexed IDs -> calibration',
                          reference_rows=len(reference_ids),calibration_rows=len(calibration_ids),
                          distinct_reference_outputs=len(reference),
                          validation_budget_pct=100*budget,
                          calibration_flagged=int(np.count_nonzero(cal>threshold)),
                          threshold=float(threshold),
                          reference_sha256=hashlib.sha256(json.dumps(dict(sorted(reference.items())),sort_keys=True).encode()).hexdigest())


def evaluate_output_collision(results: Path):
    results=results.resolve()
    raw=load_json(results/'unixcoder-effectiveness.json')
    if raw.get('schema_version')!='unixcoder-effectiveness.v1' or not raw.get('completed'):
        raise ValueError('Need completed, verified recorded UniXcoder experiment')
    selected=int(raw['best_checkpoint'].rsplit('epoch-',1)[-1])
    if selected not in {int(x['epoch']) for x in raw['history']} or selected!=max(raw['history'],key=lambda x:x['validation_bleu4'])['epoch']:
        raise ValueError('Selected checkpoint is not clean-validation BLEU maximum')
    valpath=results/f'validation-epoch-{selected}.jsonl'
    predpath=results/'poisoned-victim-triggered.jsonl'
    cleanpath=results/'poisoned-victim-clean.jsonl'
    verify_recorded_inputs(results,raw,[valpath.name,cleanpath.name,predpath.name])
    reference,fit=fit_validation_only(jsonl(valpath),.05)
    clean=unique_rows(jsonl(cleanpath),'clean test predictions')
    triggered=unique_rows(jsonl(predpath),'triggered test predictions')
    clean_ids=sorted(clean)
    tr_ids=sorted(triggered)
    if len(clean_ids)!=992 or len(tr_ids)!=416 or not set(tr_ids).issubset(clean_ids):
        raise ValueError('Unexpected held-out test protocol / unpaired IDs')
    target=normalize_output(raw['attack_target'])
    clean_scores=np.array([reference[normalize_output(clean[i]['prediction'])] for i in clean_ids],dtype=float)
    trigger_scores=np.array([reference[normalize_output(triggered[i]['prediction'])] for i in tr_ids],dtype=float)
    hits=np.array([normalize_output(triggered[i]['prediction'])==target for i in tr_ids],dtype=bool)
    clean_hits=sum(normalize_output(clean[i]['prediction'])==target for i in clean_ids)
    if (int(hits.sum())!=raw['models']['poisoned_20pct']['triggered_target_hits']
        or clean_hits!=raw['models']['poisoned_20pct']['full_clean_target_hits']):
        raise ValueError('Results disagree with original saved predictions')
    assessed=counts(clean_scores,trigger_scores,fit['threshold'],hits)
    flagged_clean=[i for i,v in zip(clean_ids,clean_scores) if v>fit['threshold']]
    flagged_triggered=[i for i,v in zip(tr_ids,trigger_scores) if v>fit['threshold']]
    flagged_target_clean=sum(normalize_output(clean[i]['prediction'])==target for i in flagged_clean)
    return {
        'schema_version':'unixcoder-output-collision-v1',
        'status':'offline_reanalysis_of_recorded_predictions',
        'novelty':'Simple output-frequency behavioral diagnostic; not a new code model or general-purpose backdoor defense',
        'model':'microsoft/unixcoder-base, recorded 20% poisoned fine-tuning',
        'selected_epoch':selected,
        'fit':fit,
        'evaluation':assessed,
        'target_hits_before_filter':int(hits.sum()),
        'clean_target_hits_before_filter':clean_hits,
        'clean_false_positives_containing_target':flagged_target_clean,
        'clean_false_positives_other_outputs':len(flagged_clean)-flagged_target_clean,
        'flagged_triggered_target_hits':sum(int(normalize_output(triggered[i]['prediction'])==target) for i in flagged_triggered),
        'flagged_triggered_non_target_hits':sum(int(normalize_output(triggered[i]['prediction'])!=target) for i in flagged_triggered),
        'example_flagged_outputs':[{'output':text, 'reference_frequency':freq}
                                   for text,freq in reference.most_common(8) if freq>fit['threshold']],
        'limitations':[
            'Only a behavioral screen of generated output text; it does NOT detect poisoned code or backdoored weights',
            'The original attack uses a fixed target phrase; paraphrased or per-input target outputs may evade this baseline',
            'Suspicious frequently generated summaries can also be legitimate; false positives are real',
            'Validation was already used for the experiment checkpoint choice; these are exploratory test-set results, not an independent generalization claim',
            'One random seed and one Python dataset subset; no new model training or fresh external benchmark',
            'Flagging a model output does not prevent its generation, only permits rejection/review after generation',
        ],
        'input_hashes':{p.name:sha256(p) for p in (valpath,cleanpath,predpath,results/'unixcoder-effectiveness.json')},
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results/unixcoder-effectiveness'))
    parser.add_argument('--output',type=Path,default=Path('results/defense-audit/output-collision.json'))
    args=parser.parse_args()
    output=evaluate_output_collision(args.results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Saved',args.output)
    x=output['evaluation']
    print('Validation reference:',output['fit']['reference_rows'],'calibration:',output['fit']['calibration_rows'],'threshold:',output['fit']['threshold'])
    print('Validation flags:',output['fit']['calibration_flagged'])
    print('Triggered detected: %d/416 (%.2f%%); clean false flags: %d/992 (%.2f%%)' % (x['true_positives'],x['recall_pct'],x['false_positives'],x['clean_fpr_pct']))
    print('Successful attacks after filtering:',x['remaining_target_hits'],'of 416 (%.2f%%)' % x['post_filter_asr_pct'])
    print('IMPORTANT: This is a fixed-output behavioral baseline, not a general backdoor detector.')

if __name__=='__main__': main()
