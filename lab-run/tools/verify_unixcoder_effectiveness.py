"""Independently recompute the completed UniXcoder report from saved case outputs.

Does not import the training runner or its metric helpers. Requires the original
saved dataset and pinned AFRAIDOOR evaluator; no GPU or model weights required.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import runpy


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): value.update(block)
    return value.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def same(actual,expected,label):
    if isinstance(expected,dict):
        for name,value in expected.items(): same(actual[name],value,label+'.'+name)
    elif isinstance(expected,float):
        if not isinstance(actual,(float,int)) or not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-9):
            raise ValueError(f'{label}: {actual} != {expected}')
    elif actual!=expected: raise ValueError(f'{label}: {actual} != {expected}')


def norm(value): return ' '.join(value.split())


def target_counts(clean,triggered,target,paired_ids):
    clean_by_id={r['index']:norm(r['prediction']) for r in clean}
    hits=sum(norm(r['prediction'])==target for r in triggered)
    paired=sum(clean_by_id[i]==target for i in paired_ids)
    total=sum(value==target for value in clean_by_id.values())
    return dict(triggered_rows=len(triggered),triggered_target_hits=hits,asr_percent=100*hits/len(triggered),
      clean_rows=len(clean),full_clean_target_hits=total,full_clean_target_percent=100*total/len(clean),
      paired_clean_target_hits=paired,net_trigger_lift_points=100*(hits-paired)/len(triggered))


def detection(negative,positive,threshold):
    from sklearn.metrics import roc_auc_score,precision_recall_curve,auc,average_precision_score
    tp=sum(s>threshold for s in positive); fp=sum(s>threshold for s in negative)
    tn=len(negative)-fp; fn=len(positive)-tp
    precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/len(positive)
    labels=[0]*len(negative)+[1]*len(positive); scores=negative+positive
    curve_p,curve_r,_=precision_recall_curve(labels,scores)
    return dict(clean_rows=len(negative),triggered_rows=len(positive),true_positives=tp,false_positives=fp,
      true_negatives=tn,false_negatives=fn,precision=precision,recall=recall,
      f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
      accuracy=(tp+tn)/len(labels),balanced_accuracy=(recall+tn/len(negative))/2,
      roc_auc=float(roc_auc_score(labels,scores)),pr_auc=float(auc(curve_r,curve_p)),
      average_precision=float(average_precision_score(labels,scores)),
      clean_false_positive_rate=fp/len(negative),trigger_detection_rate=recall)


def verify(args):
    report=json.loads((args.run/'unixcoder-effectiveness.json').read_text())
    same(report['schema_version'],'unixcoder-effectiveness.v1','schema')
    same(report['completed'],True,'completed')
    if sha(args.manifest)!=report['manifest_sha256']: raise ValueError('Manifest identity differs')
    manifest=json.loads(args.manifest.read_text())
    for relative,record in manifest['files'].items():
        if sha(args.dataset/relative)!=record['sha256']: raise ValueError('Dataset checksum: '+relative)
    for name,value in report['artifact_hashes'].items():
        if sha(args.run/name)!=value: raise ValueError('Output checksum: '+name)
    root=Path(__file__).resolve().parents[2]
    for name,value in report['source_hashes'].items():
        if sha(root/'lab-run/scripts'/name)!=value: raise ValueError('Source checksum: '+name)
    original=json.loads((args.clean/'unixcoder-clean-benchmark.json').read_text())
    same(sha(args.clean/'unixcoder-clean-benchmark.json'),report['clean_benchmark_sha256'],'clean report')
    evaluator=args.upstream/'CodeT5/evaluator/smooth_bleu.py'
    same(sha(evaluator),original['bleu_evaluator_sha256'],'evaluator')
    smooth=runpy.run_path(str(evaluator))
    clean_rows=rows(args.dataset/'clean/summarize/python/test.jsonl')
    valid_rows=rows(args.dataset/'clean/summarize/python/valid.jsonl')
    triggers=rows(args.dataset/'evaluation/test_triggered.jsonl')
    paired_ids=[r['index'] for r in triggers]
    same((len(clean_rows),len(valid_rows),len(triggers)),(992,987,416),'counts')
    def predictions(folder,prefix,expected_rows):
        values=rows(folder/(prefix+'.jsonl'))
        same([r['index'] for r in values],[r['index'] for r in expected_rows],prefix+' IDs')
        references=[' '.join(' '.join(r['docstring_tokens']).replace('_',' ').split()).replace('</s>','<unk>') for r in expected_rows]
        same([r['reference'] for r in values],references,prefix+' references')
        gold=(folder/(prefix+'.gold')).read_text().splitlines()
        same(gold,[str(r['index'])+'\t'+ref for r,ref in zip(expected_rows,references)],prefix+' gold')
        gold_map,prediction_map=smooth['computeMaps']([str(r['index'])+'\t'+norm(r['prediction']) for r in values],str(folder/(prefix+'.gold')))
        return values,smooth['bleuFromMaps'](gold_map,prediction_map)[0]
    before,control_bleu=predictions(args.clean,'clean-test',clean_rows)
    control,_=predictions(args.run,'clean-victim-triggered',triggers)
    target=norm(manifest['victim_attack_target'])
    same(report['models']['clean'],dict(target_counts(before,control,target,paired_ids),clean_bleu4=control_bleu),'clean victim')
    poisoned,poison_bleu=predictions(args.run,'poisoned-victim-clean',clean_rows)
    triggered,_=predictions(args.run,'poisoned-victim-triggered',triggers)
    same(report['models']['poisoned_20pct'],dict(target_counts(poisoned,triggered,target,paired_ids),clean_bleu4=poison_bleu),'poisoned victim')
    for record in report['history']:
        _,value=predictions(args.run,'validation-epoch-'+str(record['epoch']),valid_rows)
        same(record['validation_bleu4'],value,'validation BLEU')
    best=max(report['history'],key=lambda r:r['validation_bleu4'])['epoch']
    same(Path(report['best_checkpoint']).name,'checkpoint-epoch-'+str(best),'selection')
    frozen=json.loads((args.run/'calibration-frozen.json').read_text())
    same(frozen['checkpoint_sha256'],report['best_checkpoint_sha256'],'calibration checkpoint')
    thresholds=json.loads((args.run/'thresholds.json').read_text())
    clean_ids=[r['index'] for r in clean_rows]
    checks={}
    for method,record in report['detectors'].items():
        calibration=rows(args.run/(method+'-calibration-scores.jsonl'))
        same([r['index'] for r in calibration],[r['index'] for r in valid_rows],method+' validation IDs')
        scores=[r['score'] for r in calibration]
        if not all(math.isfinite(s) for s in scores): raise ValueError('Nonfinite scores')
        cutoff=sorted(scores)[len(scores)-math.floor(len(scores)*.05)-1]
        for value in (record['threshold'],thresholds[method],frozen['thresholds'][method]): same(value,cutoff,method+' cutoff')
        same([r['flagged'] for r in calibration],[s>cutoff for s in scores],method+' calibration flags')
        same(record['validation_flags'],sum(s>cutoff for s in scores),method+' calibration count')
        outputs=rows(args.run/(method+'-test-scores.jsonl'))
        negatives=[r for r in outputs if r['kind']=='clean']; positives=[r for r in outputs if r['kind']=='triggered']
        same(len(outputs),1408,method+' total scores')
        same([r['index'] for r in negatives],clean_ids,method+' clean score IDs')
        same([r['index'] for r in positives],paired_ids,method+' triggered score IDs')
        if not all(math.isfinite(r['score']) for r in outputs): raise ValueError('Nonfinite score')
        same([r['flagged'] for r in outputs],[r['score']>cutoff for r in outputs],method+' flags')
        negative=[r['score'] for r in negatives]; positive=[r['score'] for r in positives]
        negative_by_id={r['index']:r['score'] for r in negatives}
        same(record['metrics'],detection(negative,positive,cutoff),method+' detection')
        same(record['paired_metrics'],detection([negative_by_id[i] for i in paired_ids],positive,cutoff),method+' paired detection')
        hits=sum(norm(r['prediction'])==target and not s['flagged'] for r,s in zip(triggered,positives))
        retained=[r for r,s in zip(clean_rows,negatives) if not s['flagged']]
        kept_outputs,retained_bleu=predictions(args.run,method+'-retained-clean',retained) if retained else ([],None)
        same([r['prediction'] for r in kept_outputs],[r['prediction'] for r,s in zip(poisoned,negatives) if not s['flagged']],method+' retained predictions')
        same(record['after_filter'],dict(remaining_target_hits=hits,original_triggered_rows=416,asr_percent=100*hits/416,
          blocked_triggered=sum(r['flagged'] for r in positives),clean_accepted=len(retained),clean_rows=992,
          clean_retention_percent=100*len(retained)/992,retained_clean_bleu4=retained_bleu),method+' rejection')
        checks[method]=dict(validation_rows=987,clean_test_rows=992,triggered_test_rows=416,threshold=cutoff,
                            confusion_counts_verified=True,post_filter_original_denominator_verified=True)
    if args.checkpoint: same(sha(args.checkpoint/'adapter_model.bin'),report['best_checkpoint_sha256'],'checkpoint backup')
    verification=dict(status='independently_verified',created_utc=datetime.now(timezone.utc).isoformat(),
      report_sha256=sha(args.run/'unixcoder-effectiveness.json'),checkpoint_sha256=report['best_checkpoint_sha256'],
      checkpoint_backup_verified=bool(args.checkpoint),
      checkpoint_files={p.name:sha(p) for p in args.checkpoint.iterdir() if p.is_file()} if args.checkpoint else {},
      selected_epoch=best,models=report['models'],detectors=checks,
      prediction_references_verified=True,dataset_manifest_and_all_recorded_output_hashes_verified=True,
      verification_script_sha256=sha(__file__))
    args.output.write_text(json.dumps(verification,indent=2)+'\n')
    print('UNIXCODER_EFFECTIVENESS_INDEPENDENTLY_VERIFIED',json.dumps(verification))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','clean','dataset','manifest','upstream','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    verify(parser.parse_args())
