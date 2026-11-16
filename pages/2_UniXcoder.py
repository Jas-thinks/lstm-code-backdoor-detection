"""Read-only presentation of executed UniXcoder benchmark and detector artifacts."""
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
    st.caption('These are calibration counts. Held-out detection performance and new-model attack rates remain pending.')
    demo=FIT.parent/'detector-demo-scores.jsonl'
    if demo.is_file():
        st.write('Saved-state scoring demonstration: four clean training inputs. This is a functional check, not an effectiveness evaluation.')
        st.dataframe(pd.read_json(demo,lines=True),hide_index=True,width='stretch')
    st.download_button('Download fitting report',FIT.read_bytes(),file_name=FIT.name,mime='application/json')
except (OSError,ValueError,KeyError,TypeError) as error:
    st.info(f'Detector artifact unavailable. {error}')
st.subheader('Remaining evaluation')
st.write('Measure attack success and detector effectiveness after the two implementations are verified. The earlier 97.84% belongs to the separate CodeT5 attack experiment.')
