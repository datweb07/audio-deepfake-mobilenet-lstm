"""Distinct within-record, observational and source-matched estimand contracts."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import numpy as np

from .analysis import loss, parent_bootstrap
from .provenance import known
from .source_matching import MATCHING_VERSION


def select_matched(rows: list[dict], *, systems: tuple[str,str], minimum_sources: int=8,
                   minimum_speakers: int=5, seed: int=4804) -> dict:
    if len(set(systems)) != 2 or minimum_sources < 2 or minimum_speakers < 2:
        raise ValueError('Explicit two-system/minimum source/cluster contract required')
    grouped=defaultdict(dict)
    for row in rows:
        if row['generator_id'] in systems:
            if int(row['matching_level'])<3 or row['verification_status']!='AUTHOR_ID_MATCH':
                raise ValueError('UNKNOWN_SOURCE_IDENTITY: level 3 author evidence required')
            key=(row['language'],row['source_recording_id'])
            if row['generator_id'] in grouped[key]:
                raise ValueError('Ambiguous multiple renditions for a source/system')
            grouped[key][row['generator_id']]=row
    complete=[value for value in grouped.values() if set(value)==set(systems)]
    if len(complete)<minimum_sources:
        return {'status':'INSUFFICIENT_SOURCE_MATCHES','estimate':None,'available_sources':len(complete)}
    if len({member[systems[0]]['language'] for member in complete})!=1:
        return {'status':'LANGUAGE_CONFOUNDED','estimate':None}
    speakers={row['source_speaker_id'] for member in complete for row in member.values()}
    if any(not known(value) for value in speakers) or len(speakers)<minimum_speakers:
        return {'status':'INSUFFICIENT_INDEPENDENT_GROUPS','estimate':None,'source_text_speakers':len(speakers)}
    if minimum_sources<minimum_speakers:
        raise ValueError('Source budget cannot cover requested speaker minimum')
    ordered=sorted(complete,key=lambda value:value[systems[0]]['source_recording_id'])
    rng=np.random.default_rng(seed)
    by_speaker=defaultdict(list)
    for member in ordered:
        if len({row['source_speaker_id'] for row in member.values()})!=1:
            return {'status':'AMBIGUOUS_SOURCE_MATCH','estimate':None}
        by_speaker[member[systems[0]]['source_speaker_id']].append(member)
    queues={speaker:[members[i] for i in rng.permutation(len(members))] for speaker,members in sorted(by_speaker.items())}
    selected=[]
    while len(selected)<minimum_sources:
        for queue in queues.values():
            if queue and len(selected)<minimum_sources:
                selected.append(queue.pop())
    selected.sort(key=lambda value:value[systems[0]]['source_recording_id'])
    return {'status':'DEVELOPMENT_COHORT_IDENTIFIED','scope':'PUBLIC_DEVELOPMENT','matching_version':MATCHING_VERSION,
            'systems':list(systems),'seed':seed,'records':[member[system] for member in selected for system in systems]}


def matched_interaction(rows: list[dict], *, first_system: str, second_system: str,
                        risk: str='error', replicates: int=2000, seed: int=4804) -> dict:
    """Mean [(AB-BA)_first-(AB-BA)_second], equal source-text-speaker weight."""
    cells=defaultdict(dict)
    for row in rows:
        if row.get('scope')!='PUBLIC_DEVELOPMENT':
            raise ValueError('Confirmation/calibration data not allowed in development analysis')
        if row.get('generator_id') not in {first_system,second_system} or row.get('condition_id') not in {'AB','BA'}:
            raise ValueError('Unexpected system or factorial condition')
        if int(row.get('matching_level',0))<3 or not known(row.get('source_recording_id','UNKNOWN')):
            raise ValueError('UNKNOWN_SOURCE_IDENTITY')
        if not known(row.get('source_speaker_id','UNKNOWN')):
            raise ValueError('UNKNOWN_CLUSTER')
        if row.get('status')!='OK' or int(row['label'])!=1:
            raise ValueError('Valid FAKE-only complete scores required')
        score,threshold=float(row['raw_score']),float(row['threshold'])
        if not np.isfinite([score,threshold]).all() or not 0<=score<=1 or not 0<=threshold<=1 or int(row['predicted_label'])!=int(score>=threshold):
            raise ValueError('Invalid score or fixed-threshold rule')
        key=(row['generator_id'],row['condition_id'])
        if key in cells[row['source_recording_id']]:
            raise ValueError('Duplicate source/system/condition')
        cells[row['source_recording_id']][key]=row
    if not cells:
        return {'estimand':'SOURCE_MATCHED_SYSTEM_INTERACTION','status':'INSUFFICIENT_SOURCE_MATCHES','estimate':None}
    expected={(system,condition) for system in (first_system,second_system) for condition in ('AB','BA')}
    clusters=defaultdict(list)
    strata=set()
    for source,values in cells.items():
        if set(values)!=expected:
            return {'estimand':'SOURCE_MATCHED_SYSTEM_INTERACTION','status':'MISSING_FACTORIAL_CELL','estimate':None}
        for field in ('source_speaker_id','language','source_corpus','model_id','threshold','seed'):
            support={str(row.get(field,'UNKNOWN')) for row in values.values()}
            if len(support)!=1 or any(not known(value) for value in support):
                return {'status':'CONFOUNDED_FACTOR_'+field,'estimate':None}
        if any(values[(system,'AB')]['sample_id']!=values[(system,'BA')]['sample_id'] for system in (first_system,second_system)):
            raise ValueError('Within-record order pairing mismatch')
        strata.add(tuple(str(next(iter(values.values()))[key]) for key in ('model_id','threshold','seed')))
        contrast=sum(coefficient*loss(values[(system,condition)],risk) for system,condition,coefficient in
                     ((first_system,'AB',1),(first_system,'BA',-1),(second_system,'AB',-1),(second_system,'BA',1)))
        clusters[next(iter(values.values()))['source_speaker_id']].append(contrast)
    if len(strata)!=1:
        raise ValueError('Mixed model/threshold/seed strata')
    if len(clusters)<2:
        return {'status':'INSUFFICIENT_CLUSTERS','estimate':None}
    return {'estimand':'SOURCE_MATCHED_SYSTEM_INTERACTION','scope':'DESCRIPTIVE_DEVELOPMENT',
            'status':'INSUFFICIENT_INDEPENDENT_GROUPS' if len(clusters)<5 else 'DESCRIPTIVE_SOURCE_MATCHED',
            'risk':risk,'source_recordings':len(cells),'cluster_unit':'original source-text speaker',
            **parent_bootstrap(clusters,replicates=replicates,seed=seed),'causal_interpretation':False}
