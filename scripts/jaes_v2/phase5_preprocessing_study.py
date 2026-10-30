"""Paired deployment-feasible P1/P2 diagnostic with existing frozen detector weights."""
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.preprocessing import load_audio
from src.lava.jaes_v2.evidence import file_hash,write_csv,write_json,read_csv
from src.lava.jaes_v2.preprocessing import Preprocessor
from src.lava.jaes_v2.phase3 import verify_seal
from src.lava.jaes_v2.runtime import LegacyMelPredictor,preflight,source_fingerprint
from src.lava.jaes_v2.public_analysis import metrics


def main() -> None:
    root=Path(__file__).resolve().parents[2]
    output=root/'outputs/jaes_v2/phase5_preprocessing_study_v1'
    if output.exists():
        raise FileExistsError('New study directory required')
    environment=preflight(root)
    cache=root/'outputs/jaes_v2/c1_nuisance_controls_dev_v1_waveforms'
    verify_seal(cache,json.loads((cache/'cache_metadata.json').read_text()))
    old=root/'outputs/jaes_v2/c1_mlaad_tiny_dev_v1_waveforms'
    verify_seal(old,json.loads((old/'cache_metadata.json').read_text()))
    manifest=read_csv(cache/'manifest.csv'); ids={r['sample_id'] for r in manifest}
    paths={(r['sample_id'],r['condition_id']):old/r['path'] for r in read_csv(old/'conditions.csv') if r['sample_id'] in ids}
    output.mkdir(parents=True)
    write_csv(output/'manifest.csv',manifest)
    protocol={'scope':'EXPLORATORY_PREPROCESSOR_DISTRIBUTION_SHIFT','ids':sorted(ids),'policies':[asdict(Preprocessor(policy=p)) for p in ('P0','P1','P2')],
              'gains':[1,.5],'model_ids':['mobilenetv3_lstm','shufflenetv2_lstm'],'thresholds':'INHERITED','training':'NONE',
              'P3':'Amplitude training augmentation, not an inference preprocessor; evaluated separately in controlled training',
              'manifest_sha256':file_hash(cache/'manifest.csv')}
    write_json(output/'protocol.json',protocol)
    metadata={'status':'RUNNING','environment':environment,'code':source_fingerprint(root),'source_hashes':{},'artifacts':{},'timings':{}}
    write_json(output/'run_metadata.json',metadata)
    rows=[]; differences=[]; started=perf_counter()
    for model in protocol['model_ids']:
        predictor=LegacyMelPredictor(model)
        import tensorflow as tf
        layer=next(layer for layer in predictor.detector.model.layers if isinstance(layer,tf.keras.layers.LSTM))
        probe=tf.keras.Model(predictor.detector.model.inputs,[predictor.detector.model.output,layer.output])
        metadata['artifacts'][model]=predictor.artifact_hashes
        # Original P0 scores were already executed in Phase 4. Do not repeat those scored forwards.
        prior=root/'outputs/jaes_v2'/('c1_nuisance_controls_dev_v1_'+model)
        previous=read_csv(prior/'scores.csv'); verify_seal(prior,json.loads((prior/'run_metadata.json').read_text()))
        for source in manifest:
            sample=source['sample_id']; path=paths[(sample,'clean')]
            metadata['source_hashes'][str(path)]=file_hash(path)
            native=load_audio(path)
            for policy in ('P0','P1','P2'):
                preprocessor=Preprocessor(policy=policy)
                features=[preprocessor.extract(native*gain) for gain in protocol['gains']]
                scores=[]; embeddings=[]
                for gain,feature in zip(protocol['gains'],features):
                    begin=perf_counter()
                    score,embedding=probe(tf.convert_to_tensor(feature[None]),training=False)
                    score=float(score.numpy().reshape(-1)[0]); vector=embedding.numpy().reshape(-1)
                    # P0 probe calls are embedding diagnostics; compare to sealed scores, never call them new evaluation evidence.
                    if policy=='P0':
                        member='phase3_baseline' if gain==1 else 'global_half_v1'
                        reference=next(r for r in previous if r['sample_id']==sample and r['policy']==member and r['condition_id']=='clean')
                        if abs(score-float(reference['raw_score']))>1e-5:
                            raise ValueError('P0 reference score/embedding bridge parity failed')
                    scores.append(score); embeddings.append(vector)
                    rows.append({'sample_id':sample,'label':source['label'],'speaker_id':source['source_speaker_id'],'model_id':model,
                                 'preprocessor_id':policy,'preprocessor_hash':preprocessor.fingerprint,'condition_id':'native' if gain==1 else 'half',
                                 'raw_score':score,'threshold':predictor.threshold,'predicted_label':int(score>=predictor.threshold),
                                 'embedding_l2':float(np.linalg.norm(vector)),'embedding_dimension':len(vector),
                                 'forward_ms':(perf_counter()-begin)*1000,'evidence_role':'P0_BRIDGE_PARITY_AND_EMBEDDING' if policy=='P0' else 'NEW_PREPROCESSOR_DIAGNOSTIC',
                                 'checkpoint_hash':next(iter(predictor.artifact_hashes.values()))})
                differences.append({'sample_id':sample,'model_id':model,'policy':policy,'feature_max_difference':float(np.max(np.abs(features[1]-features[0]))),
                                    'feature_mean_difference':float(np.mean(np.abs(features[1]-features[0]))),
                                    'embedding_l2_difference':float(np.linalg.norm(embeddings[1]-embeddings[0])),
                                    'raw_score_difference':scores[1]-scores[0],
                                    'decision_flip':int((scores[0]>=predictor.threshold)!=(scores[1]>=predictor.threshold))})
            write_csv(output/'scores.csv',rows); write_csv(output/'paired_differences.csv',differences)
        print(f'{model}: completed {len(manifest)*6} preprocessing/embedding forwards',flush=True)
        tf.keras.backend.clear_session()
    table=[]
    for model in protocol['model_ids']:
        for policy in ('P0','P1','P2'):
            for condition in ('native','half'):
                selected=[r for r in rows if r['model_id']==model and r['preprocessor_id']==policy and r['condition_id']==condition]
                table.append({'model':model,'policy':policy,'condition':condition,**metrics(selected)})
    write_csv(output/'metrics.csv',table)
    write_json(output/'statistics.json',{'scope':protocol['scope'],'metrics':table,'pairs':differences,
              'limitations':['No-retraining changed-preprocessor distribution shift, not trained-model quality','32 development records / four known FAKE source-text speakers',
                             'Silence policy and numerical floor differ; preserved gain invariance need not preserve discrimination']})
    metadata.update(status='DEVELOPMENT_DIAGNOSTIC_EVALUATED',elapsed_seconds=perf_counter()-started,forward_calls=len(rows),
                    new_policy_forward_calls=sum(r['preprocessor_id']!='P0' for r in rows),P0_embedding_parity_calls=sum(r['preprocessor_id']=='P0' for r in rows))
    metadata['output_hashes']={p.relative_to(output).as_posix():file_hash(p) for p in output.rglob('*') if p.is_file() and p.name!='run_metadata.json'}
    write_json(output/'run_metadata.json',metadata)


if __name__=='__main__':
    main()
