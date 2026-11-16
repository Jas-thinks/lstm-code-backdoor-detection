"""Fixed-budget UniXcoder attack and defense evaluation.

Reuses the recorded AFRAIDOOR LSTM-crafted 20% dataset. Selects the victim
checkpoint on clean validation BLEU, then fits detectors on training features
and calibrates on clean validation. Test labels never select model or cutoff.
"""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
from benchmark_detection import PARTITIONS,UPSTREAM_REVISION,digest,write_json,validate_partitions
from benchmark_pretrained import load_and_audit

SCHEMA='unixcoder-effectiveness.v1'
EXPECTED=dict(train=9897,valid=987,test=992,triggered=416,poisoned=1979)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def attack_metrics(clean_rows,triggered_rows,before,after,target):
    if len(clean_rows)!=len(before) or len(triggered_rows)!=len(after) or not before or not after:
        raise ValueError('Incomplete predictions')
    norm=lambda value:' '.join(value.split())
    target=norm(target)
    clean={row['index']:norm(value) for row,value in zip(clean_rows,before)}
    indices=[row['index'] for row in triggered_rows]
    if len(clean)!=len(clean_rows) or len(set(indices))!=len(indices) or any(i not in clean for i in indices):
        raise ValueError('Duplicate or unpaired evaluation IDs')
    hits=sum(norm(value)==target for value in after)
    paired=sum(clean[i]==target for i in indices)
    total=sum(value==target for value in clean.values())
    return dict(triggered_rows=len(after),triggered_target_hits=hits,asr_percent=100*hits/len(after),
                clean_rows=len(before),full_clean_target_hits=total,full_clean_target_percent=100*total/len(before),
                paired_clean_target_hits=paired,net_trigger_lift_points=100*(hits-paired)/len(after))


def filter_metrics(clean_scores,trigger_scores,threshold,predictions,target):
    import numpy as np
    clean=np.asarray(clean_scores); triggered=np.asarray(trigger_scores)
    if len(triggered)!=len(predictions) or not len(clean) or not len(triggered): raise ValueError('Incomplete filter inputs')
    accepted=triggered<=threshold
    hits=sum(' '.join(value.split())==' '.join(target.split()) and bool(keep) for value,keep in zip(predictions,accepted))
    return dict(remaining_target_hits=hits,original_triggered_rows=len(triggered),asr_percent=100*hits/len(triggered),
                blocked_triggered=int((~accepted).sum()),clean_accepted=int((clean<=threshold).sum()),clean_rows=len(clean),
                clean_retention_percent=100*int((clean<=threshold).sum())/len(clean),
                policy='Reject flagged inputs; original triggered denominator retained')


def validate_data(manifest,dataset,prepared,counts=EXPECTED):
    if (manifest['seed'],manifest['train_rows'],manifest['valid_rows'],manifest['clean_test_rows'],
        manifest['triggered_test_rows'],manifest['poisoned_train_rows'])!=(42,counts['train'],counts['valid'],counts['test'],counts['triggered'],counts['poisoned']):
        raise ValueError('Use the recorded seed42 20% dataset')
    parts={}
    for name,relative in PARTITIONS.items():
        path=dataset/relative
        if digest(path)!=manifest['files'][relative]['sha256']: raise ValueError('Dataset checksum differs: '+relative)
        rows=read_rows(path)
        if len(rows)!=manifest['files'][relative]['rows'] or len({r['index'] for r in rows})!=len(rows):
            raise ValueError('Dataset length or unique IDs differ: '+relative)
        parts[name]=rows
    validate_partitions(parts,counts['poisoned'])
    for split,name in (('train','clean_train'),('valid','clean_validation'),('test','clean_test')):
        if len(parts[name])!=len(prepared[split]): raise ValueError('Prepared split length differs')
        for original,row in zip(prepared[split],parts[name]):
            if (original['index']!=row['index'] or original['sha256_hash']!=row['original_sha256_hash'] or
                original['code_tokens']!=row['code_tokens'] or original['docstring_tokens']!=row['docstring_tokens']):
                raise ValueError('Clean input differs from the completed benchmark')
    target=manifest['victim_attack_target']
    for clean,row in zip(parts['clean_train'],parts['mixed_train']):
        if row['poison']:
            if row['docstring_tokens']!=target.split() or row.get('original_docstring_tokens')!=clean['docstring_tokens']:
                raise ValueError('Invalid poisoned target/reference')
            if not row['trigger_applied'] or row['code_tokens']==clean['code_tokens']: raise ValueError('Missing training trigger')
        if hashlib.sha256(row['source_code'].encode()).hexdigest()!=row['sha256_hash']: raise ValueError('Mixed source identity differs')
    for rows in parts.values():
        for row in rows:
            if hashlib.sha256(row['source_code'].encode()).hexdigest()!=row['sha256_hash']:
                raise ValueError('Source content checksum differs')
            if row.get('attack_target')!=target: raise ValueError('Attack target metadata differs')
    if not all(row.get('trigger_applied') for row in parts['triggered_test']): raise ValueError('Missing test trigger')
    return parts


def train_victim(parts,settings,args,smooth):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset,DataLoader
    from transformers import get_linear_schedule_with_warmup
    from huggingface_hub import hf_hub_download
    from unixcoder_adapter import UniXcoderSummarizer,MODEL_ID,MODEL_REVISION,PRETRAINED_SHA256
    from train_codet5 import source_text,reference_text,score_predictions
    import random
    random.seed(settings['seed']); np.random.seed(settings['seed']); torch.manual_seed(settings['seed'])
    weights=hf_hub_download(MODEL_ID,'pytorch_model.bin',revision=MODEL_REVISION)
    if digest(weights)!=PRETRAINED_SHA256: raise ValueError('Pretrained weight checksum mismatch')
    model=UniXcoderSummarizer.from_pretrained().float()
    source=model.tokenize([source_text(r) for r in parts['mixed_train']],length=settings['max_source_length'])
    target=model.tokenize([reference_text(r) for r in parts['mixed_train']],length=settings['max_target_length'],target=True)
    loader=DataLoader(TensorDataset(source,target),batch_size=settings['batch_size'],shuffle=True,
                      generator=torch.Generator().manual_seed(settings['seed']))
    model.cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=0)
    accumulation=settings['accumulation']; epochs=settings['epochs']
    scheduler=get_linear_schedule_with_warmup(optimizer,settings['warmup_steps'],epochs*((len(loader)+accumulation-1)//accumulation))
    best=-1.; selected=None; history=[]
    for epoch in range(1,epochs+1):
        model.train(); optimizer.zero_grad(); window=0; total=0.; supervised=0
        for step,(s,t) in enumerate(loader,1):
            t=t.cuda(); count=int(t[:,1:].ne(model.pad).sum())
            loss,_=model(s.cuda(),t)
            if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
            (loss*count).backward(); window+=count; supervised+=count; total+=float(loss.detach())*count
            if step%accumulation==0 or step==len(loader):
                for parameter in model.parameters():
                    if parameter.grad is not None: parameter.grad.div_(window)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(); window=0
            if step%100==0: print('UNIXCODER_POISON_TRAIN',epoch,step,len(loader),flush=True)
        predictions=predict(model,parts['clean_validation'],settings)
        bleu=score_predictions(parts['clean_validation'],predictions,smooth,args.out/f'validation-epoch-{epoch}')
        history.append(dict(epoch=epoch,train_nll=total/supervised,validation_bleu4=bleu))
        if bleu>best:
            best=bleu; selected=args.out/f'checkpoint-epoch-{epoch}'; model.save(selected)
        write_json(args.out/'history.json',history)
        print('UNIXCODER_POISON_VALIDATION',json.dumps(history[-1]),flush=True)
    return selected,history


def predict(model,rows,settings):
    from train_codet5 import source_text
    model.eval(); predictions=[]
    for start in range(0,len(rows),settings['batch_size']):
        tokens=model.tokenize([source_text(r) for r in rows[start:start+settings['batch_size']]],length=settings['max_source_length'])
        predictions.extend(model.summarize(tokens,max_new_tokens=settings['max_target_length']))
    return predictions


def evaluate(args,parts,clean,plan,smooth,selected,history):
    import numpy as np
    import torch
    from unixcoder_adapter import UniXcoderSummarizer
    from unixcoder_detection import features
    from feature_detectors import fit_detectors,score_features,fit_threshold,detection_metrics
    from train_codet5 import score_predictions,source_text
    target=plan['attack_target']; settings=plan['settings']
    result=dict(plan,history=history,best_checkpoint=str(selected.resolve()),best_checkpoint_sha256=digest(selected/'adapter_model.bin'),models={},detectors={})
    # Clean control: retain existing per-case clean predictions, generate triggered inputs now.
    if digest(args.clean_checkpoint/'adapter_model.bin')!=clean['best_checkpoint_sha256']: raise ValueError('Clean checkpoint differs')
    predictions=read_rows(args.clean_report.parent/'clean-test.jsonl')
    if [r['index'] for r in predictions]!=[r['index'] for r in parts['clean_test']]: raise ValueError('Clean prediction alignment differs')
    model=UniXcoderSummarizer.load(args.clean_checkpoint).float().cuda().eval()
    print('UNIXCODER_CLEAN_CONTROL_STARTED',flush=True)
    after=predict(model,parts['triggered_test'],settings)
    score_predictions(parts['triggered_test'],after,smooth,args.out/'clean-victim-triggered')
    result['models']['clean']=dict(attack_metrics(parts['clean_test'],parts['triggered_test'],[r['prediction'] for r in predictions],after,target),clean_bleu4=clean['clean_test_bleu4'])
    del model; gc.collect(); torch.cuda.empty_cache()
    model=UniXcoderSummarizer.load(selected).float().cuda().eval()
    # Freeze fitting/calibration before scoring held-out examples.
    print('UNIXCODER_POISONED_DETECTOR_FIT_STARTED',flush=True)
    train=features(model,parts['clean_train'],8)
    changed=[i for i,(a,b) in enumerate(zip(parts['clean_train'],parts['mixed_train'])) if a['code_tokens']!=b['code_tokens']]
    if len(changed)!=1979: raise ValueError('Wrong changed training count')
    mixed=train.copy(); mixed[changed]=features(model,[parts['mixed_train'][i] for i in changed],8)
    states=fit_detectors(train,mixed)
    valid=features(model,parts['clean_validation'],8); thresholds={}
    for method,state in states.items():
        calibration=score_features(state,valid)
        thresholds[method]=fit_threshold(calibration,.05)
        with (args.out/(method+'-calibration-scores.jsonl')).open('w') as stream:
            for row,score in zip(parts['clean_validation'],calibration):
                stream.write(json.dumps(dict(index=row['index'],score=float(score),flagged=bool(score>thresholds[method])))+'\n')
        np.savez_compressed(args.out/(method+'.npz'),**state)
    write_json(args.out/'thresholds.json',thresholds)
    write_json(args.out/'calibration-frozen.json',dict(status='thresholds_frozen_before_test_features',checkpoint_sha256=result['best_checkpoint_sha256'],
      thresholds=thresholds,source_hashes=plan['source_hashes']))
    print('UNIXCODER_THRESHOLDS_FROZEN_TEST_STARTED',flush=True)
    before=predict(model,parts['clean_test'],settings); after=predict(model,parts['triggered_test'],settings)
    bleu=score_predictions(parts['clean_test'],before,smooth,args.out/'poisoned-victim-clean')
    score_predictions(parts['triggered_test'],after,smooth,args.out/'poisoned-victim-triggered')
    result['models']['poisoned_20pct']=dict(attack_metrics(parts['clean_test'],parts['triggered_test'],before,after,target),clean_bleu4=bleu)
    clean_features=features(model,parts['clean_test'],8); triggered_features=features(model,parts['triggered_test'],8)
    positions={r['index']:i for i,r in enumerate(parts['clean_test'])}
    paired_positions=[positions[r['index']] for r in parts['triggered_test']]
    # Count trigger changes visible after the new victim's own tokenization/truncation.
    visible=0
    for a,b in zip(parts['paired_clean_test'],parts['triggered_test']):
        ids=model.tokenize([source_text(a),source_text(b)],length=256)
        visible+=not torch.equal(ids[0],ids[1])
    result['tokenization_audit']=dict(triggered_rows=416,visible_input_changes=visible,asr_denominator_unchanged=416)
    for method,state in states.items():
        threshold=thresholds[method]; negative=score_features(state,clean_features); positive=score_features(state,triggered_features)
        kept=[i for i,value in enumerate(negative) if value<=threshold]
        retained_bleu=score_predictions([parts['clean_test'][i] for i in kept],[before[i] for i in kept],smooth,args.out/(method+'-retained-clean')) if kept else None
        result['detectors'][method]=dict(threshold=threshold,validation_flags=int((score_features(state,valid)>threshold).sum()),
          metrics=detection_metrics(negative,positive,threshold),paired_metrics=detection_metrics(negative[paired_positions],positive,threshold),
          after_filter=dict(filter_metrics(negative,positive,threshold,after,target),retained_clean_bleu4=retained_bleu))
        with (args.out/(method+'-test-scores.jsonl')).open('w') as stream:
            for kind,rows,scores in (('clean',parts['clean_test'],negative),('triggered',parts['triggered_test'],positive)):
                for row,score in zip(rows,scores):
                    stream.write(json.dumps(dict(index=row['index'],kind=kind,score=float(score),flagged=bool(score>threshold)))+'\n')
    result['completed']=True
    result['completed_utc']=datetime.now(timezone.utc).isoformat()
    result['artifact_hashes']={p.name:digest(p) for p in args.out.iterdir() if p.is_file()}
    write_json(args.out/'unixcoder-effectiveness.json',result)
    print('UNIXCODER_EFFECTIVENESS_COMPLETE',json.dumps(dict(models=result['models'],detectors=result['detectors'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','dataset-dir','data-dir','upstream','clean-report','clean-checkpoint','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if args.out.exists(): raise FileExistsError('Choose a fresh run folder')
    prepared,audit=load_and_audit(args.data_dir)
    manifest=json.loads(args.manifest.read_text()); parts=validate_data(manifest,args.dataset_dir,prepared)
    clean=json.loads(args.clean_report.read_text())
    if clean.get('completed') is not True or clean['schema_version']!='unixcoder-clean-benchmark.v1': raise ValueError('Completed clean benchmark required')
    if clean['model_id']!='microsoft/unixcoder-base' or clean['model_revision']!='5604afdc964f6c53782a6813140ade5216b99006': raise ValueError('Wrong reference model')
    expected_settings=dict(seed=42,epochs=3,batch_size=4,accumulation=8,learning_rate=5e-5,max_source_length=256,max_target_length=128,beam_size=1,warmup_steps=100,scheduler='linear',precision='float32')
    if clean['settings']!=expected_settings: raise ValueError('Reference training protocol differs')
    reference=Path(__file__).resolve().parents[2]/'results/unixcoder-clean/verification.json'
    verified=json.loads(reference.read_text())
    if digest(args.clean_report.parent/'clean-test.jsonl')!=verified['artifact_hashes']['clean-test.jsonl']:
        raise ValueError('Saved clean-control predictions differ from published evidence')
    if clean['best_checkpoint_sha256']!=verified['checkpoint_sha256']: raise ValueError('Wrong clean reference checkpoint')
    for split in ('train','valid','test'):
        if audit['splits'][split]['decompressed_sha256']!=clean['data_audit']['splits'][split]['decompressed_sha256']: raise ValueError('Clean reference changed')
    revision=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()
    evaluator=args.upstream/'CodeT5/evaluator/smooth_bleu.py'
    original=subprocess.check_output(['git','-C',str(args.upstream),'show',UPSTREAM_REVISION+':CodeT5/evaluator/smooth_bleu.py'])
    if revision!=UPSTREAM_REVISION or digest(evaluator)!=hashlib.sha256(original).hexdigest(): raise ValueError('Evaluator changed')
    plan=dict(schema_version=SCHEMA,completed=False,created_utc=datetime.now(timezone.utc).isoformat(),
      model_id=clean['model_id'],model_revision=clean['model_revision'],settings=clean['settings'],data_audit=audit,
      poison_budget=dict(rows=1979,training_rows=9897,rate=1979/9897),attack_target=manifest['victim_attack_target'],
      manifest_sha256=digest(args.manifest),input_hashes={name:manifest['files'][relative]['sha256'] for name,relative in PARTITIONS.items()},
      clean_benchmark_sha256=digest(args.clean_report),source_hashes={name:digest(Path(__file__).parent/name) for name in
      ('unixcoder_effectiveness.py','unixcoder_adapter.py','unixcoder_detection.py','feature_detectors.py')},
      protocol=dict(selection='Highest clean-validation BLEU; no ASR selection',generation='Fixed one beam, no sampling',
        detector_calibration='Clean validation only; fixed empirical 5% FPR budget; strict greater cutoff',
        spectral_reference='Unlabeled mixed training',mahalanobis_reference='Trusted original clean training',
        test='992 clean,416 triggered,416 matched clean; established reused test set',
        generalization='One seed, Python, one trigger family, fixed20% budget; transfer of saved LSTM-crafted inputs',
        defense='Reject flagged inputs; report original416 denominator and clean retention'))
    args.out.mkdir(parents=True)
    write_json(args.out/'protocol.json',plan)
    if args.prepare_only: print('UNIXCODER_EFFECTIVENESS_PREPARED',args.out); return
    import torch
    import transformers
    import numpy as np
    import sklearn
    from benchmark_detection import gpu_lock
    if not torch.cuda.is_available(): raise RuntimeError('Use the lab CUDA environment')
    if torch.__version__!='1.13.1+cu117' or transformers.__version__!='4.26.1': raise RuntimeError('Use the pinned project environment')
    plan['runtime']=dict(torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,sklearn=sklearn.__version__,gpu=torch.cuda.get_device_name(0))
    write_json(args.out/'protocol.json',plan)
    with gpu_lock():
        smooth=runpy.run_path(str(evaluator))
        selected,history=train_victim(parts,plan['settings'],args,smooth)
        gc.collect(); torch.cuda.empty_cache()
        evaluate(args,parts,clean,plan,smooth,selected,history)

if __name__=='__main__': main()
