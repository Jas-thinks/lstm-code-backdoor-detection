import gzip, json, os, runpy
from pathlib import Path
import torch
import train_codet5 as core
from adaptive_triggers import Crafter, Ineligible
from prepare_poisoned_dataset import victim_record

root = Path.cwd() / 'run'
repo, project = root / 'upstream', root / 'baseline'
report = json.loads((project / 'latest_poisoned_dataset.json').read_text())
trained = json.loads((project / 'latest_poisoned_victim.json').read_text())
checkpoint = Path(trained['best_checkpoint'])
assert core.digest(checkpoint / 'pytorch_model.bin') == trained['checkpoint_weights_sha256']
target = report['victim_attack_target']
settings = trained['settings']
out = root / 'decode-sweep'
out.mkdir(exist_ok=True)
normalize = lambda text: ' '.join(text.split())
assert torch.cuda.is_available()
crafter = Crafter(repo, project)
clean, triggered = [], []
with gzip.open(repo / 'datasets/csn_python_prepared/normalized/valid.jsonl.gz', 'rt') as rows:
    for line in rows:
        row = json.loads(line)
        clean.append(victim_record(row, target))
        if normalize(core.reference_text(clean[-1])) == target:
            continue
        try:
            plan = crafter.plan(row)
        except Ineligible:
            continue
        triggered.append(victim_record(row, target, crafter.craft(row, plan)))
print('Validation clean and triggered counts', len(clean), len(triggered), flush=True)
del crafter
torch.cuda.empty_cache()
tokenizer = core.RobertaTokenizer.from_pretrained(checkpoint, local_files_only=True)
model = core.T5ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True).cuda().eval()
smooth = runpy.run_path(str(repo / 'CodeT5/evaluator/smooth_bleu.py'))

def predict(rows, beams):
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            batch = rows[start:start + 16]
            tokens = tokenizer([core.source_text(row) for row in batch], padding='max_length', truncation=True,
                               max_length=settings['max_source_length'], return_tensors='pt').to('cuda')
            generated = model.generate(**tokens, num_beams=beams, max_length=settings['max_target_length'],
                                       early_stopping=beams > 1, do_sample=False, use_cache=True)
            predictions.extend(tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False))
    return predictions

def evaluate(clean_rows, trigger_rows, beams, split):
    before = predict(clean_rows, beams)
    after = predict(trigger_rows, beams)
    before_by_id = {row['index']: normalize(prediction) for row, prediction in zip(clean_rows, before)}
    hits = sum(normalize(prediction) == target for prediction in after)
    paired_clean_hits = sum(before_by_id[row['index']] == target for row in trigger_rows)
    clean_hits = sum(value == target for value in before_by_id.values())
    bleu = core.score_predictions(clean_rows, before, smooth, out / f'{split}-beam-{beams}')
    result = dict(beams=beams, hits=hits, count=len(trigger_rows), asr=100 * hits / len(trigger_rows),
                  clean_bleu=bleu, clean_target_rate=100 * clean_hits / len(clean_rows),
                  paired_clean_hits=paired_clean_hits, net_trigger_lift=100 * (hits-paired_clean_hits) / len(trigger_rows))
    print(split, json.dumps(result), flush=True)
    return result

validation = [evaluate(clean, triggered, beams, 'validation') for beams in (5, 1, 3, 8)]
baseline = validation[0]
eligible = [x for x in validation if x['clean_bleu'] >= baseline['clean_bleu'] - 1.0
            and x['clean_target_rate'] <= baseline['clean_target_rate'] + 2.0]
selected = max(eligible, key=lambda x: (x['asr'], x['net_trigger_lift'], x['clean_bleu']))
core.write_json(out / 'selection.json', dict(selected=selected, validation=validation))
print('Selected beams', selected['beams'], flush=True)

def load_data(relative):
    path = Path(report['dataset_dir']) / relative
    assert core.digest(path) == report['files'][relative]['sha256']
    return [json.loads(line) for line in path.read_text().splitlines()]

if selected['beams'] != 5:
    result = evaluate(load_data('clean/summarize/python/test.jsonl'),
                      load_data('evaluation/test_triggered.jsonl'), selected['beams'], 'test')
else:
    result = json.loads((root / 'logs/team-experiment-results.json').read_text())['baseline']
core.write_json(out / 'final.json', dict(selected_beams=selected['beams'], test=result))
print('DECODE_SWEEP_COMPLETE', json.dumps(result), flush=True)
