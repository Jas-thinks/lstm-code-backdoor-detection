import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path.cwd() / 'run'
project = root / 'poison-budget-10pct'
scripts = Path.cwd() / 'scripts'
sys.path.insert(0, str(scripts))
source = (scripts / 'decode_sweep.py').read_text()
setup, sep, _ = source.partition('\nvalidation =')
assert sep and "root / 'baseline'" in setup and "root / 'decode-sweep'" in setup
setup = setup.replace("root / 'baseline'", "root / 'poison-budget-10pct'").replace("root / 'decode-sweep'", "project / 'evaluation'")
ns = {'__file__': str(scripts / 'decode_sweep.py')}
exec(compile(setup, str(scripts / 'decode_sweep.py'), 'exec'), ns)
core, tokenizer, normalize = ns['core'], ns['tokenizer'], ns['normalize']
report, predict, target = ns['report'], ns['predict'], ns['target']
limit = ns['settings']['max_source_length']

def load(relative):
    path = Path(report['dataset_dir']) / relative
    assert core.digest(path) == report['files'][relative]['sha256'], relative
    return [json.loads(line) for line in path.read_text().splitlines()]

def distance(a,b):
    row = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(row[j]+1, nxt[-1]+1, row[j-1]+(x!=y)))
        row = nxt
    return row[-1]

def analyze(clean, triggered, name):
    assert len({r['index'] for r in clean}) == len(clean)
    clean_by_id = {r['index']:r for r in clean}
    paired = [clean_by_id[r['index']] for r in triggered]
    clean_text = [core.source_text(r) for r in paired]
    trigger_text = [core.source_text(r) for r in triggered]
    before = predict(paired, 1)
    after = predict(triggered, 1)
    full_before = tokenizer(clean_text, truncation=False, verbose=False)['input_ids']
    full_after = tokenizer(trigger_text, truncation=False, verbose=False)['input_ids']
    capped_before = tokenizer(clean_text, truncation=True, max_length=limit)['input_ids']
    capped_after = tokenizer(trigger_text, truncation=True, max_length=limit)['input_ids']
    buckets = defaultdict(Counter)
    categories = Counter()
    details = []
    for row, c, t, fb, fa, cb, ca in zip(triggered, before, after, full_before, full_after, capped_before, capped_after):
        c, t = normalize(c), normalize(t)
        hit = t == target
        stats = buckets['hit' if hit else 'miss']
        stats['total'] += 1
        stats['source_over_limit'] += len(fa) > limit
        stats['same_encoded_input'] += cb == ca
        stats['trigger_removed_by_truncation'] += cb == ca and fb != fa
        stats['same_as_clean_output'] += t == c
        stats['blank_output'] += not t
        words = distance(t.split(), target.split())
        stats['within_two_word_edits'] += 0 < words <= 2
        if not hit:
            if cb == ca and fb != fa:
                label = 'trigger_removed_by_truncation'
            elif cb == ca:
                label = 'no_input_change_after_tokenization'
            elif 0 < words <= 2:
                label = 'near_target'
            elif c == t:
                label = 'same_as_clean_output'
            else:
                label = 'different_non_target_output'
            categories[label] += 1
            details.append(dict(index=row['index'], category=label, target_word_edit_distance=words,
                                clean_prediction=c, triggered_prediction=t, full_input_tokens=len(fa), encoded_inputs_equal=cb==ca))
    result = dict(split=name, source_token_limit=limit, groups={k:dict(v) for k,v in buckets.items()}, miss_categories=dict(categories),
                  frequent_miss_predictions=Counter(d['triggered_prediction'] for d in details).most_common(6),
                  near_match_examples=[d['triggered_prediction'] for d in details if d['category']=='near_target'][:3])
    print('MISS_AUDIT_'+name.upper(),json.dumps(result),flush=True)
    return result, details

validation, val_details = analyze(ns['clean'], ns['triggered'], 'validation')
assert validation['groups']['hit']['total'] == 299
heldout, test_details = analyze(load('clean/summarize/python/test.jsonl'),load('evaluation/test_triggered.jsonl'),'test')
assert heldout['groups']['hit']['total'] == 315 and heldout['groups']['miss']['total'] == 101
assert sum(heldout['miss_categories'].values()) == 101
core.write_json(project / 'miss-audit.json',dict(validation=validation, test=heldout))
core.write_json(project / 'miss-audit-cases.json',dict(validation=val_details, test=test_details))
print('MISS_AUDIT_COMPLETE',flush=True)
