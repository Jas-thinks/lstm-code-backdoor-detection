"""Fine-tune the new UniXcoder victim on clean data, select by validation BLEU.

This stage measures summarization quality only. It performs no attack-rate
optimization. Use --prepare-only to audit data and save the GPU run plan locally.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import runpy
import subprocess
from benchmark_detection import digest, write_json, UPSTREAM_REVISION, gpu_lock
from benchmark_pretrained import load_and_audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model-dir',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args()
    if args.out.exists(): raise FileExistsError('Use a fresh output folder')
    rows,audit=load_and_audit(args.data_dir)
    revision=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()
    if revision!=UPSTREAM_REVISION: raise ValueError('Unexpected upstream evaluator revision')
    evaluator=args.upstream/'CodeT5/evaluator/smooth_bleu.py'
    import hashlib
    original=subprocess.check_output(['git','-C',str(args.upstream),'show',UPSTREAM_REVISION+':CodeT5/evaluator/smooth_bleu.py'])
    if digest(evaluator)!=hashlib.sha256(original).hexdigest(): raise ValueError('Modified BLEU evaluator')
    settings=dict(seed=42,epochs=3,batch_size=4,accumulation=8,learning_rate=5e-5,
                  max_source_length=256,max_target_length=128,beam_size=1,warmup_steps=100,
                  scheduler='linear',precision='float32')
    plan=dict(schema_version='unixcoder-clean-benchmark.v1',completed=False,
              created_utc=datetime.now(timezone.utc).isoformat(),model_id='microsoft/unixcoder-base',
              model_revision='5604afdc964f6c53782a6813140ade5216b99006',settings=settings,data_audit=audit,
              bleu_evaluator_sha256=digest(evaluator),
              selection='Highest clean validation BLEU; all test inputs held until checkpoint selection',
              protocol='New model; one-beam prefix-LM adaptation; CodeT5 results are references, not identical training protocols',
              implementation_hashes={name:digest(Path(__file__).parent/name) for name in ('train_unixcoder.py','unixcoder_adapter.py')})
    args.out.mkdir(parents=True)
    write_json(args.out/'protocol.json',plan)
    if args.prepare_only:
        print('UNIXCODER_PLAN_PREPARED',args.out); return
    import random
    import numpy as np
    import torch
    from contextlib import nullcontext
    from torch.utils.data import TensorDataset,DataLoader
    from transformers import get_linear_schedule_with_warmup
    from unixcoder_adapter import UniXcoderSummarizer,MODEL_ID,MODEL_REVISION,PRETRAINED_SHA256
    from huggingface_hub import hf_hub_download
    from train_codet5 import source_text,reference_text,score_predictions
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('GPU unavailable; use --prepare-only')
    with gpu_lock() if args.device=='cuda' else nullcontext():
        weights=args.model_dir/'pytorch_model.bin' if args.model_dir else Path(hf_hub_download(MODEL_ID,'pytorch_model.bin',revision=MODEL_REVISION))
        if digest(weights)!=PRETRAINED_SHA256: raise ValueError('UniXcoder pretrained weights checksum mismatch')
        model=UniXcoderSummarizer.from_pretrained(args.model_dir or MODEL_ID,local_files_only=bool(args.model_dir))
        import transformers
        plan['runtime']=dict(torch=torch.__version__,transformers=transformers.__version__,device=args.device,pretrained_weights_sha256=PRETRAINED_SHA256)
        source=model.tokenize([source_text(r) for r in rows['train']],length=256)
        target=model.tokenize([reference_text(r) for r in rows['train']],length=128,target=True)
        loader=DataLoader(TensorDataset(source,target),batch_size=4,shuffle=True,
                          generator=torch.Generator().manual_seed(42))
        model.to(args.device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=0)
        scheduler=get_linear_schedule_with_warmup(optimizer,100,3*((len(loader)+7)//8))
        smooth=runpy.run_path(str(args.upstream/'CodeT5/evaluator/smooth_bleu.py'))
        def evaluate(split,prefix):
            predictions=[]; model.eval()
            for i in range(0,len(rows[split]),4):
                tokens=model.tokenize([source_text(r) for r in rows[split][i:i+4]],length=256)
                predictions.extend(model.summarize(tokens,max_new_tokens=128))
            return score_predictions(rows[split],predictions,smooth,args.out/prefix)
        best=-1; selected=None; history=[]
        for epoch in range(1,4):
            model.train(); optimizer.zero_grad(); window=0; total=0.; supervised=0
            for step,(s,t) in enumerate(loader,1):
                t=t.to(args.device); count=int(t[:,1:].ne(model.pad).sum())
                loss,_=model(s.to(args.device),t)
                if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
                (loss*count).backward(); window+=count; supervised+=count; total+=float(loss.detach())*count
                if step%8==0 or step==len(loader):
                    for parameter in model.parameters():
                        if parameter.grad is not None: parameter.grad.div_(window)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step(); scheduler.step(); optimizer.zero_grad(); window=0
                if step%100==0: print('UNIXCODER_TRAIN',epoch,step,len(loader),flush=True)
            bleu=evaluate('valid',f'validation-epoch-{epoch}')
            history.append(dict(epoch=epoch,train_nll=total/supervised,validation_bleu4=bleu))
            if bleu>best:
                best=bleu; selected=args.out/f'checkpoint-epoch-{epoch}'; model.save(selected)
            write_json(args.out/'history.json',history)
        model=UniXcoderSummarizer.load(selected).to(args.device).eval()
        test_bleu=evaluate('test','clean-test')
        plan.update(completed=True,history=history,best_checkpoint=str(selected.resolve()),
                    best_checkpoint_sha256=digest(selected/'adapter_model.bin'),
                    clean_test_rows=len(rows['test']),clean_test_bleu4=test_bleu)
        write_json(args.out/'unixcoder-clean-benchmark.json',plan)
        print('UNIXCODER_CLEAN_BENCHMARK_COMPLETE',test_bleu,flush=True)

if __name__=='__main__': main()
