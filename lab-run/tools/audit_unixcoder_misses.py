"""Describe held-out target misses from saved predictions, without retuning."""
import argparse,json
from pathlib import Path


def normalized(value): return ' '.join(value.split())


def word_distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1): current.append(min(previous[j]+1,current[-1]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    report=json.loads((args.run/'unixcoder-effectiveness.json').read_text())
    if report.get('completed') is not True: raise ValueError('Complete measurement first')
    read=lambda name:[json.loads(s) for s in (args.run/name).read_text().splitlines()]
    target=normalized(report['attack_target'])
    clean={r['index']:r for r in read('poisoned-victim-clean.jsonl')}
    triggered=read('poisoned-victim-triggered.jsonl');misses=[]
    for row in triggered:
        prediction=normalized(row['prediction']);before=normalized(clean[row['index']]['prediction'])
        if prediction!=target:
            misses.append(dict(index=row['index'],prediction=prediction,paired_clean_prediction=before,
                reference=row['reference'],same_output_as_clean=prediction==before,
                word_edits_to_target=word_distance(prediction.split(),target.split())))
    hits=len(triggered)-len(misses)
    if hits!=report['models']['poisoned_20pct']['triggered_target_hits']:raise ValueError('Hit accounting differs')
    result=dict(status='descriptive_saved_prediction_audit',triggered_rows=len(triggered),target_hits=hits,misses=len(misses),
        unchanged_output_misses=sum(r['same_output_as_clean'] for r in misses),
        changed_non_target_misses=sum(not r['same_output_as_clean'] for r in misses),
        within_two_word_edits=sum(r['word_edits_to_target']<=2 for r in misses),cases=misses,
        limits='Describes outputs only; no claim about causal failure mechanism or gradient provenance; no test-based retuning')
    (args.run/'miss-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))

if __name__=='__main__': main()
