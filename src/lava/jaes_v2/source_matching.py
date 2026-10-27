"""Evidence-ranked source matching; flattened basenames never certify lineage."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from enum import IntEnum
from itertools import combinations
from pathlib import Path, PurePosixPath
import unicodedata

from .evidence import object_hash, read_csv, write_csv, write_json
from .mlaad import metadata_inventory
from .provenance import known

MATCHING_VERSION = 'author-full-source-id-and-text-1'


class MatchLevel(IntEnum):
    UNKNOWN = 0
    SOURCE_SPEAKER = 1
    DOCUMENTED_TEXT = 2
    AUTHOR_SOURCE_ID = 3
    VERIFIED_ORIGINAL_CONTENT = 4


def text_key(value: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', value).split())


def relationship(first: dict, second: dict) -> dict:
    """A level-4 claim requires explicit independently supplied original-content evidence."""
    a, b = first.get('source_recording_id','UNKNOWN'), second.get('source_recording_id','UNKNOWN')
    ta, tb = text_key(first.get('transcript','')), text_key(second.get('transcript',''))
    level, status, evidence = MatchLevel.UNKNOWN, 'UNKNOWN_SOURCE_RELATIONSHIP', 'NONE'
    if known(a) and known(b) and a == b:
        if ta and tb and ta != tb:
            return dict(matching_level=0, verification_status='CONFLICTING_AUTHOR_TEXT', matching_evidence='same full original_file but different transcripts')
        level, status, evidence = MatchLevel.AUTHOR_SOURCE_ID, 'AUTHOR_ID_MATCH', 'identical full author original_file'
        if ta and tb:
            evidence += '; identical normalized author transcript'
        # No automatic escalation from text, local byte hash, or acoustic similarity.
    elif known(a) and known(b) and PurePosixPath(a).name == PurePosixPath(b).name:
        status, evidence = 'AMBIGUOUS_SOURCE_MATCH', 'same basename with different full source paths; not merged'
    elif ta and tb and ta == tb:
        level, status, evidence = MatchLevel.DOCUMENTED_TEXT, 'AUTHOR_TEXT_MATCH_ONLY', 'exact normalized author text; original recordings differ/unknown'
    elif known(first.get('source_speaker_id','UNKNOWN')) and first.get('source_speaker_id') == second.get('source_speaker_id'):
        level, status, evidence = MatchLevel.SOURCE_SPEAKER, 'SPEAKER_ONLY', 'same original source-text speaker; different recordings'
    return dict(matching_level=int(level), verification_status=status, matching_evidence=evidence)


def enriched_inventory(discovery: Path) -> tuple[list[dict], dict]:
    rows, summary = metadata_inventory(discovery)  # verifies every author metadata hash first
    annotations = {}
    for path in sorted(discovery.rglob('meta.csv')):
        with path.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream,delimiter='|'):
                annotations[row['path'].removeprefix('./')] = row
    for row in rows:
        annotation = annotations.get(row['path'],{})
        row['transcript'] = annotation.get('transcript','')
        row['author_reference_speaker'] = annotation.get('reference_speaker','UNKNOWN') or 'UNKNOWN'
        row['matching_version'] = MATCHING_VERSION
    return rows,summary


def pair_overlap(rows: list[dict]) -> list[dict]:
    groups = defaultdict(set)
    for row in rows:
        if int(row['label']) == 1 and known(row.get('source_recording_id','UNKNOWN')):
            groups[row['generator_id']].add(row['source_recording_id'])
    return [{'G0':a,'G1':b,'source_overlap':len(groups[a]&groups[b]),'sources_G0':len(groups[a]),'sources_G1':len(groups[b])}
            for a,b in combinations(sorted(groups),2)]


def audit_matching(discovery: Path, selected_manifest: Path, output: Path) -> dict:
    output.mkdir(parents=True,exist_ok=False)
    rows,summary = enriched_inventory(discovery)
    selected = read_csv(selected_manifest)
    selected_pairs = pair_overlap(selected)
    fake = [row for row in rows if row['label'] == 1]
    by_source, basenames = defaultdict(list), defaultdict(set)
    for row in fake:
        source = row['source_recording_id']
        if known(source):
            by_source[(row['language'],source)].append(row)
            basenames[PurePosixPath(source).name].add(source)
    candidates,verified,rejected,matrix = [],[],[],[]
    for (language,source),members in sorted(by_source.items()):
        generators = sorted({row['generator_id'] for row in members})
        matrix.append({'language':language,'source_recording_id':source,'generator_count':len(generators),
                       'generator_ids':'|'.join(generators),'recordings':len(members)})
        if len(generators) < 2:
            continue
        texts = {text_key(row['transcript']) for row in members if row['transcript']}
        for row in members:
            match = relationship(row,next(other for other in members if other['generator_id'] != row['generator_id']))
            entry = {key:row[key] for key in ('source_recording_id','source_speaker_id','generator_id','sample_id','language','path')}
            entry.update(match,author_transcript_sha256=object_hash(text_key(row['transcript'])),matching_version=MATCHING_VERSION)
            candidates.append(entry)
            if len(texts) > 1:
                rejected.append({**entry,'verification_status':'CONFLICTING_AUTHOR_TEXT','matching_level':0})
            elif match['matching_level'] >= 3:
                verified.append(entry)
            else:
                rejected.append(entry)
    real_candidates=[]
    for row in rows:
        if row['label']:
            continue
        matches=sorted(basenames.get(PurePosixPath(row['path']).name,set()))
        real_candidates.append({'sample_id':row['sample_id'],'path':row['path'],'candidate_original_ids':'|'.join(matches),
                                'matching_level':0,'verification_status':'AMBIGUOUS_SOURCE_MATCH' if len(matches)>1 else
                                'UNVERIFIED_BASENAME_CANDIDATE' if matches else 'NO_DOCUMENTED_MAPPING'})
    fields=['source_recording_id','source_speaker_id','generator_id','sample_id','language','path','matching_level',
            'verification_status','matching_evidence','author_transcript_sha256','matching_version']
    write_csv(output/'source_match_candidates.csv',candidates,fields)
    write_csv(output/'source_match_verified.csv',verified,fields)
    write_csv(output/'source_match_rejected.csv',rejected,fields)
    write_csv(output/'source_generator_matrix.csv',matrix)
    write_csv(output/'real_fake_mapping_candidates.csv',real_candidates)
    write_csv(output/'phase3_pairwise_overlap.csv',selected_pairs)
    english=[row for row in fake if row['language']=='en']
    pairs=pair_overlap(english)
    write_csv(output/'all_english_pairwise_overlap.csv',pairs)
    systems=[]
    for group in sorted({row['generator_id'] for row in english}):
        members=[row for row in english if row['generator_id']==group]
        systems.append({'system':group,'available_recordings':len(members),'unique_source_ids':len({r['source_recording_id'] for r in members}),
                        'source_text_speakers':len({r['source_speaker_id'] for r in members}),
                        'cross_generator_source_ids':len({r['source_recording_id'] for r in verified if r['generator_id']==group and r['language']=='en'}),
                        'matching_confidence':'LEVEL_3_AUTHOR_SOURCE_ID; not original-content verification'})
    write_csv(output/'table1_source_feasibility.csv',systems)
    report={'version':MATCHING_VERSION,'revision':summary['revision'],'phase3_pairwise_overlap':selected_pairs,
            'fake_recordings':len(fake),'unique_source_ids':len({r['source_recording_id'] for r in fake}),
            'author_system_names':len({r['generator_id'] for r in fake}),
            'source_text_speakers':len({r['source_speaker_id'] for r in fake}-{'UNKNOWN'}),
            'same_language_multisystem_sources':sum(row['generator_count']>=2 for row in matrix),
            'same_language_source_multiplicity':dict(Counter(str(row['generator_count']) for row in matrix)),
            'verified_candidate_rows':len(verified),'rejected_candidate_rows':len(rejected),
            'verified_level4_sources':0,'real_verified_source_mappings':0,
            'real_basename_collisions':sum(row['verification_status']=='AMBIGUOUS_SOURCE_MATCH' for row in real_candidates),
            'max_english_pair_overlap':max(row['source_overlap'] for row in pairs),
            'best_english_pairs':sorted(pairs,key=lambda row:(-row['source_overlap'],row['G0'],row['G1']))[:10],
            'status':'SOURCE_MATCHED_EXPERIMENT_BLOCKED',
            'reason':'Maximum common sources per English system pair below prespecified pilot minimum 8; only four known English source-text speakers',
            'population_independent_units':'UNKNOWN'}
    write_json(output/'matching_coverage.json',report)
    return report
