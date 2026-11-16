"""Apply verified phase-three detector states to new code-token JSONL inputs.

Inference only: no fitting, threshold selection or effectiveness-rate claims.
"""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
from benchmark_detection import digest,gpu_lock
from detect_codet5 import load_inputs

METHODS=('spectral_signature','shrinkage_mahalanobis')


def load_artifacts(folder,checkpoint):
    report_path=folder/'unixcoder-effectiveness.json'
    report=json.loads(report_path.read_text())
    if report.get('schema_version')!='unixcoder-effectiveness.v1' or report.get('completed') is not True:
        raise ValueError('Completed phase-three evidence required')
    verification=json.loads((folder/'verification.json').read_text())
    if verification.get('status')!='independently_verified' or verification.get('report_sha256')!=digest(report_path):
        raise ValueError('Independent report verification required')
    if verification.get('checkpoint_backup_verified') is not True or not verification.get('checkpoint_files'):
        raise ValueError('Checkpoint backup verification required')
    if digest(checkpoint/'adapter_model.bin')!=report['best_checkpoint_sha256']:
        raise ValueError('Checkpoint weights checksum differs')
    for name,value in verification['checkpoint_files'].items():
        if Path(name).name!=name or digest(checkpoint/name)!=value: raise ValueError('Checkpoint file checksum differs')
    for name,value in report['source_hashes'].items():
        if Path(name).name!=name or digest(Path(__file__).parent/name)!=value: raise ValueError('Feature implementation checksum differs')
    import numpy as np
    thresholds=json.loads((folder/'thresholds.json').read_text())
    if digest(folder/'thresholds.json')!=report['artifact_hashes']['thresholds.json']:
        raise ValueError('Saved threshold file checksum differs')
    states={}
    for method in METHODS:
        value=thresholds[method]
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value!=report['detectors'][method]['threshold']:
            raise ValueError('Threshold differs from completed report')
        path=folder/(method+'.npz')
        if digest(path)!=report['artifact_hashes'][path.name]: raise ValueError('Detector state checksum differs')
        with np.load(path,allow_pickle=False) as saved: states[method]=dict(saved)
        if str(states[method]['method'])!=method: raise ValueError('Detector method differs')
    return report,thresholds,states


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts','checkpoint','input','out'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda'); p.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args()
    if args.out.exists(): raise FileExistsError('Choose a fresh output filename')
    if args.batch_size<1: p.error('Use a positive batch size')
    rows=load_inputs(args.input)
    report,thresholds,states=load_artifacts(args.artifacts,args.checkpoint)
    import torch
    from unixcoder_adapter import UniXcoderSummarizer
    from unixcoder_detection import features
    from feature_detectors import score_features
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    with gpu_lock() if args.device=='cuda' else nullcontext():
        model=UniXcoderSummarizer.load(args.checkpoint).float().to(args.device).eval()
        values=features(model,rows,args.batch_size)
        scores={method:score_features(state,values) for method,state in states.items()}
    with args.out.open('x') as stream:
        for i,row in enumerate(rows):
            stream.write(json.dumps(dict(index=row['index'],detectors={method:dict(score=float(value[i]),threshold=thresholds[method],
                flagged=bool(value[i]>thresholds[method])) for method,value in scores.items()}),allow_nan=False)+'\n')
    print(json.dumps(dict(rows_scored=len(rows),output=str(args.out),checkpoint_sha256=report['best_checkpoint_sha256'],
                         policy='Independent flags at saved cutoffs; no refitting or rate measurement')))

if __name__=='__main__': main()
