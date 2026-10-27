"""Metadata-only review of confirmation drafts; never opens audio or labels."""
from __future__ import annotations

from .provenance import known


def assess_confirmation(draft: dict, *, development_ids: set[str]) -> dict:
    missing=[]
    for field in ('dataset','revision','license_evidence','primary_hypothesis','primary_estimand','factor_definitions',
                  'cluster_evidence','matching_rule','model_artifact_hashes','threshold_policy','conditions',
                  'precision_evidence','multiplicity_policy','success_failure_criteria','source_overlap_audit'):
        if not draft.get(field):
            missing.append(field)
    identities=draft.get('sample_ids',[])
    if not identities or len(set(identities))!=len(identities):
        missing.append('unique_exact_confirmation_ids')
    if set(identities)&development_ids:
        missing.append('development_confirmation_disjointness')
    if draft.get('verified_independent_clusters',0)<20:
        missing.append('sufficient_verified_independent_clusters')
    if draft.get('validation_kind')=='A_SOURCE_MATCHED_HYPOTHESIS' and draft.get('matching_level',0)<3:
        missing.append('authoritative_source_content_support')
    if draft.get('validation_kind') not in {'A_SOURCE_MATCHED_HYPOTHESIS','B_CROSS_CORPUS_TRANSFER'}:
        missing.append('validation_kind')
    if draft.get('confirmation_labels_opened',True):
        missing.append('untouched_confirmation_labels')
    return {'status':'CONFIRMATION_NOT_LOCKED' if missing else 'DRAFT_COMPLETE_PROVENANCE_REVIEW_REQUIRED',
            'missing_requirements':sorted(set(missing)),'validation_kind':draft.get('validation_kind','UNKNOWN'),
            'confirmation_execution_enabled':False,'labels_read':False,
            'interpretation':'Validation B never certifies Validation A; this checker does not authenticate evidence or open a confirmation cohort'}
