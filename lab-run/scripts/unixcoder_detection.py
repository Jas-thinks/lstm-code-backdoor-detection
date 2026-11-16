"""Fit/calibrate or apply a UniXcoder detector without effectiveness scoring.

Training features fit OAS covariance; clean validation sets the cutoff. Test
features and attack labels never enter fitting. Scoring does not change state.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from benchmark_detection import digest,write_json,gpu_lock
from benchmark_pretrained import load_and_audit
from detect_codet5 import load_inputs

SCHEMA='unixcoder-detector-fit.v1'


def features(model,rows,batch_size):
    import numpy as np
    import torch
    from train_codet5 import source_text
    arrays=[]
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            ids=model.tokenize([source_text(row) for row in rows[start:start+batch_size]],length=256)
            arrays.append(model.features(ids).cpu().numpy())
    if not arrays: raise ValueError('No input rows')
    values=np.concatenate(arrays)
    if not np.isfinite(values).all(): raise ValueError('Non-finite features')
    return values


def verify_fit(folder):
    report=json.loads((folder/'detector-fit.json').read_text())
    if report.get('schema_version')!=SCHEMA or report.get('completed') is not True:
        raise ValueError('A complete fitting artifact is required')
    import math
    if not isinstance(report.get('threshold'),(int,float)) or not math.isfinite(report['threshold']):
        raise ValueError('Invalid saved threshold')
    for name,expected in report['artifact_hashes'].items():
        if Path(name).name!=name or digest(folder/name)!=expected: raise ValueError('Detector artifact checksum mismatch')
    threshold=json.loads((folder/'threshold.json').read_text())['threshold']
    if threshold!=report['threshold']: raise ValueError('Saved thresholds differ')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    fit=sub.add_parser('fit')
    fit.add_argument('--benchmark-report',type=Path,required=True)
    fit.add_argument('--data-dir',type=Path,required=True)
    fit.add_argument('--checkpoint',type=Path,required=True)
    fit.add_argument('--out',type=Path,required=True)
    score=sub.add_parser('score')
    score.add_argument('--artifacts',type=Path,required=True)
    score.add_argument('--checkpoint',type=Path,required=True)
    score.add_argument('--input',type=Path,required=True)
    score.add_argument('--out',type=Path,required=True)
    for parser in (fit,score):
        parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
        parser.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args()
    if args.out.exists(): raise FileExistsError('Choose a new output path')
    if args.batch_size<1: p.error('Use a positive batch size')
    if args.action=='fit':
        benchmark=json.loads(args.benchmark_report.read_text())
        if benchmark.get('schema_version')!='unixcoder-clean-benchmark.v1' or benchmark.get('completed') is not True:
            raise ValueError('Complete the new-model clean benchmark before detector fitting')
        expected=benchmark['best_checkpoint_sha256']
        rows,audit=load_and_audit(args.data_dir)
        for split in ('train','valid'):
            if audit['splits'][split]['decompressed_sha256']!=benchmark['data_audit']['splits'][split]['decompressed_sha256']:
                raise ValueError('Detector input differs from benchmark')
    else:
        report=verify_fit(args.artifacts); expected=report['checkpoint_sha256']
        rows=load_inputs(args.input)
    if digest(args.checkpoint/'adapter_model.bin')!=expected: raise ValueError('Checkpoint checksum mismatch')
    import numpy as np
    import torch
    from unixcoder_adapter import UniXcoderSummarizer
    from feature_detectors import fit_clean_reference,fit_threshold,score_features
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if args.action=='score':
        for name,expected_hash in report['checkpoint_files'].items():
            if Path(name).name!=name or digest(args.checkpoint/name)!=expected_hash:
                raise ValueError('Checkpoint tokenizer/configuration checksum mismatch')
    with gpu_lock() if args.device=='cuda' else nullcontext():
        model=UniXcoderSummarizer.load(args.checkpoint).float().to(args.device).eval()
        if args.action=='fit':
            train=features(model,rows['train'],args.batch_size)
            state=fit_clean_reference(train)
            valid=score_features(state,features(model,rows['valid'],args.batch_size))
            threshold=fit_threshold(valid,0.05)
            args.out.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(args.out/'shrinkage_mahalanobis.npz',**state)
            write_json(args.out/'threshold.json',dict(threshold=threshold,rule='score > threshold',max_validation_fpr=0.05))
            with (args.out/'calibration-scores.jsonl').open('x') as stream:
                for row,value in zip(rows['valid'],valid):
                    stream.write(json.dumps(dict(index=row['index'],score=float(value),flagged=bool(value>threshold)))+'\n')
            report=dict(schema_version=SCHEMA,completed=True,status='fitted_and_calibrated_not_effectiveness_evaluated',
                        method='shrinkage_mahalanobis',model_role='clean_finetuned_unixcoder',checkpoint_sha256=expected,
                        model_id=benchmark['model_id'],model_revision=benchmark['model_revision'],
                        covariance_shrinkage=float(state['shrinkage']),
                        pooling='Attention-mask mean of final bidirectional source representations including mode tokens',
                        reference_hashes={split:audit['splits'][split]['decompressed_sha256'] for split in ('train','valid')},
                        runtime=dict(torch=torch.__version__,numpy=np.__version__,device=args.device,batch_size=args.batch_size),
                        checkpoint_files={path.name:digest(path) for path in args.checkpoint.iterdir() if path.suffix=='.json' or path.name=='merges.txt'},
                        train_rows=len(train),validation_rows=len(valid),feature_dimension=train.shape[1],
                        threshold=threshold,validation_flags=int((valid>threshold).sum()),max_validation_fpr=0.05,
                        trusted_reference='Clean training data; clean validation calibration; no test representations used',
                        benchmark_report_sha256=digest(args.benchmark_report),
                        source_hashes={name:digest(Path(__file__).parent/name) for name in ('unixcoder_detection.py','unixcoder_adapter.py','feature_detectors.py')},
                        artifact_hashes={name:digest(args.out/name) for name in ('shrinkage_mahalanobis.npz','threshold.json','calibration-scores.jsonl')})
            write_json(args.out/'detector-fit.json',report)
            print('UNIXCODER_DETECTOR_FIT_COMPLETE',args.out,flush=True)
        else:
            with np.load(args.artifacts/'shrinkage_mahalanobis.npz',allow_pickle=False) as archive: state=dict(archive)
            scores=score_features(state,features(model,rows,args.batch_size))
            threshold=report['threshold']
            with args.out.open('x') as stream:
                for row,value in zip(rows,scores):
                    stream.write(json.dumps(dict(index=row['index'],score=float(value),threshold=threshold,flagged=bool(value>threshold)))+'\n')
            print('UNIXCODER_INPUTS_SCORED',len(rows),flush=True)

if __name__=='__main__': main()
