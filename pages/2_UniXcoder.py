"""Read-only presentation of executed UniXcoder benchmark and detector artifacts."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/'results/unixcoder-clean/unixcoder-clean-benchmark.json'
FIT=ROOT/'results/unixcoder-detector/detector-fit.json'
st.set_page_config(page_title='UniXcoder extension',layout='wide')
st.title('UniXcoder: new-model extension')
st.write('A model outside AFRAIDOOR’s CodeBERT, PLBART and CodeT5 trio, evaluated on Python code summarization.')
try:
    report=json.loads(BENCH.read_text())
    if report.get('schema_version')!='unixcoder-clean-benchmark.v1' or report.get('completed') is not True:
        raise ValueError('The clean benchmark is incomplete.')
    if report.get('model_id')!='microsoft/unixcoder-base': raise ValueError('Unexpected model.')
    selected=max(report['history'],key=lambda r:r['validation_bleu4'])
    columns=st.columns(3)
    columns[0].metric('Clean test BLEU-4',f"{report['clean_test_bleu4']:.4f}")
    columns[1].metric('Clean test inputs',report['clean_test_rows'])
    columns[2].metric('Selected epoch',selected['epoch'])
    st.caption('This measures summary quality. It is not an attack-success or detection-accuracy percentage.')
    st.subheader('Training and checkpoint selection')
    st.dataframe(pd.DataFrame(report['history']).rename(columns={
        'epoch':'Epoch','train_nll':'Training token NLL','validation_bleu4':'Clean validation BLEU-4'}),
        hide_index=True,width='stretch')
    st.write('All three epochs were retained in the record. Highest validation BLEU selected the checkpoint before test evaluation.')
    chart=BENCH.parent/'training-history.png'
    if chart.is_file(): st.image(str(chart))
    st.download_button('Download benchmark report',BENCH.read_bytes(),file_name=BENCH.name,mime='application/json')
except (OSError,ValueError,KeyError,TypeError) as error:
    st.info(f'A completed benchmark report is required. {error}')
    st.stop()

st.subheader('Additional detector implementation')
try:
    fit=json.loads(FIT.read_text())
    if fit.get('schema_version')!='unixcoder-detector-fit.v1' or fit.get('completed') is not True:
        raise ValueError('Detector fitting has not completed.')
    st.write('OAS shrinkage Mahalanobis distance flags representations that are outliers against a trusted clean reference.')
    st.write(f"Fitted on {fit['train_rows']:,} clean training examples; calibrated on {fit['validation_rows']:,} clean validation examples; {fit['feature_dimension']} features.")
    st.write(f"Saved cutoff: {fit['threshold']:.6f}. Calibration flags: {fit['validation_flags']}/{fit['validation_rows']} within the empirical 5% budget.")
    st.caption('This initial calibration belongs to the clean model. Effectiveness uses fresh detector states under the poisoned model below.')
    demo=FIT.parent/'detector-demo-scores.jsonl'
    if demo.is_file():
        st.write('Saved-state scoring demonstration: four clean training inputs. This is a functional check, not an effectiveness evaluation.')
        st.dataframe(pd.read_json(demo,lines=True),hide_index=True,width='stretch')
    st.download_button('Download fitting report',FIT.read_bytes(),file_name=FIT.name,mime='application/json')
except (OSError,ValueError,KeyError,TypeError) as error:
    st.info(f'Detector artifact unavailable. {error}')
st.subheader('Attack and defense effectiveness')
EFFECT=ROOT/'results/unixcoder-effectiveness/unixcoder-effectiveness.json'
try:
    measured=json.loads(EFFECT.read_text())
    verified=json.loads((EFFECT.parent/'verification.json').read_text())
    if measured.get('schema_version')!='unixcoder-effectiveness.v1' or measured.get('completed') is not True:
        raise ValueError('Effectiveness evaluation is incomplete.')
    if verified.get('status')!='independently_verified' or verified.get('report_sha256')!=hashlib.sha256(EFFECT.read_bytes()).hexdigest():
        raise ValueError('The report requires independent evidence verification.')
    attack=[]
    for name,model in measured['models'].items():
        attack.append({'Victim':'Clean control' if name=='clean' else '20% poisoned',
          'Triggered target hits':f"{model['triggered_target_hits']}/{model['triggered_rows']}",
          'Attack success (%)':model['asr_percent'],'Clean target (%)':model['full_clean_target_percent'],
          'Paired clean target hits':model['paired_clean_target_hits'],
          'Net trigger lift (pp)':model['net_trigger_lift_points'],'Clean BLEU-4':model['clean_bleu4']})
    st.dataframe(pd.DataFrame(attack),hide_index=True,width='stretch')
    st.write('The same 416 triggered inputs are used for both victims. Checkpoint selection uses clean validation BLEU. Target matching uses the complete recorded summary.')
    defense=[]
    labels={'spectral_signature':'Spectral signature','shrinkage_mahalanobis':'OAS Mahalanobis'}
    for name,detector in measured['detectors'].items():
        metrics=detector['metrics']; after=detector['after_filter']
        defense.append({'Detector':labels.get(name,name),'Trigger recall (%)':100*metrics['recall'],
          'Clean false positives (%)':100*metrics['clean_false_positive_rate'],'Precision (%)':100*metrics['precision'],
          'F1 (%)':100*metrics['f1'],'ROC AUC':metrics['roc_auc'],'PR AUC':metrics['pr_auc'],
          'Post-filter target hits':f"{after['remaining_target_hits']}/{after['original_triggered_rows']}",
          'Post-filter ASR (%)':after['asr_percent'],'Clean retention (%)':after['clean_retention_percent'],
          'Retained-clean BLEU-4':after['retained_clean_bleu4']})
    st.dataframe(pd.DataFrame(defense),hide_index=True,width='stretch')
    st.caption('Lower post-filter ASR is desirable. Rejected attacks remain in the original denominator; clean retention shows the cost of filtering. The 5% budget is an empirical validation limit, not a guarantee on test false positives.')
    audit=measured['tokenization_audit']
    st.write(f"UniXcoder tokenization retains an input change on {audit['visible_input_changes']}/{audit['triggered_rows']} triggered inputs. The ASR denominator stays {audit['asr_denominator_unchanged']}.")
    for name in ('effectiveness-summary.png','detector-roc.png'):
        if (EFFECT.parent/name).is_file(): st.image(str(EFFECT.parent/name))
    st.download_button('Download effectiveness report',EFFECT.read_bytes(),file_name=EFFECT.name,mime='application/json')
    st.write('Scope: one seed, Python subset, fixed 20% poison budget, transferred saved LSTM-crafted inputs and the established test set. OAS assumes trusted clean training data. These results are team evidence; each member should review their assigned artifacts.')
except (OSError,ValueError,KeyError,TypeError) as error:
    st.info(f'Verified effectiveness results are not yet available. {error}')
st.caption('The earlier 97.84% is the separate CodeT5 attack result.')
