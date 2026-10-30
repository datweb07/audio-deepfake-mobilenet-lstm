"""Development-only grouped metadata diagnostics; constant features are noninformative."""
from __future__ import annotations

from collections import defaultdict
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .provenance import known


def available_source_clusters(manifest: list[dict]) -> list[str]:
    """Connected known source-speaker and parent components; UNKNOWN never joins sources."""
    parents = list(range(len(manifest)))
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    seen = {}
    for index,row in enumerate(manifest):
        for field in ('source_speaker_id','speaker_id','source_recording_id','parent_recording_id'):
            value = row.get(field,'UNKNOWN')
            if known(value):
                key = (field,value)
                if key in seen:
                    parents[find(index)] = find(seen[key])
                seen[key] = index
    return [str(find(index)) for index in range(len(manifest))]


def grouped_shortcuts(features: list[dict], manifest: list[dict], *, seed: int = 3804) -> dict:
    if not manifest or any(row['split'] != 'development' for row in manifest):
        raise ValueError('Shortcut fitting is restricted to development; test/calibration labels prohibited')
    indexed = {row['sample_id']:row for row in features}
    if set(indexed) != {row['sample_id'] for row in manifest} or len(indexed) != len(features):
        raise ValueError('Feature/manifest identity mismatch')
    labels = np.array([int(row['label']) for row in manifest])
    groups = np.array(available_source_clusters(manifest))
    if set(labels) != {0,1}:
        raise ValueError('Both classes required')
    splits = list(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=seed).split(np.zeros(len(labels)),labels,groups))
    if any(set(labels[train]) != {0,1} for train,_ in splits):
        raise ValueError('Insufficient grouped training coverage')
    keys = ('duration','sample_rate','silence_fraction_below_1e-4','rms','peak')
    definitions = {'duration_only':('duration',),'sample_rate_only':('sample_rate',),
                   'silence_only':('silence_fraction_below_1e-4',),'rms_only':('rms',),'numeric_combined':keys}
    results = []
    for name,selected in definitions.items():
        matrix = np.array([[float(indexed[row['sample_id']][key]) for key in selected] for row in manifest])
        if not np.isfinite(matrix).all():
            raise ValueError('Nonfinite shortcut feature')
        if not np.any(np.ptp(matrix,axis=0)):
            results.append({'baseline':name,'accuracy':None,'dummy_accuracy':None,'status':'NONINFORMATIVE_CONSTANT_FEATURE'})
            continue
        predicted, dummy = np.empty(len(labels),dtype=int),np.empty(len(labels),dtype=int)
        for train,validation in splits:
            model = make_pipeline(StandardScaler(),LogisticRegression(C=1,solver='liblinear',random_state=0))
            model.fit(matrix[train],labels[train])
            predicted[validation] = model.predict(matrix[validation])
            control = DummyClassifier(strategy='most_frequent').fit(matrix[train],labels[train])
            dummy[validation] = control.predict(matrix[validation])
        results.append({'baseline':name,'accuracy':float(np.mean(predicted==labels)),
                        'dummy_accuracy':float(np.mean(dummy==labels)),'status':'EXPLORATORY_GROUPED_DEVELOPMENT'})
    return {'scope':'PUBLIC_DEVELOPMENT_ONLY','recordings':len(labels),'available_source_components':len(set(groups)),
            'folds':[{'train':len(train),'heldout':len(test),'heldout_fake':int(labels[test].sum())} for train,test in splits],
            'grouping':'Connected known original source-speaker/parent components; REAL unknown lineage remains local',
            'population_independence':'UNKNOWN','no_hyperparameter_selection':True,'results':results}
