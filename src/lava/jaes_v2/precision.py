"""Development-cluster variance planning; never reads a confirmation cohort."""
from __future__ import annotations

from math import ceil, sqrt
import numpy as np
from scipy.stats import chi2, t


def plan_precision(values: dict[str,list[float]], *, target_effect: float=.10, half_width: float=.05) -> dict:
    if len(values)<2 or any(not group for group in values.values()) or not 0<target_effect<=1 or not 0<half_width<1:
        raise ValueError('At least two nonempty development clusters and valid targets required')
    groups=[np.asarray(values[key],dtype=np.float64) for key in sorted(values)]
    if any(not np.isfinite(group).all() for group in groups):
        raise ValueError('Nonfinite planning data')
    means=np.array([group.mean() for group in groups]); k=len(groups); n=sum(map(len,groups))
    variance=float(means.var(ddof=1)); critical=float(t.ppf(.975,k-1))
    lower=(k-1)*variance/chi2.ppf(.975,k-1)
    upper=(k-1)*variance/chi2.ppf(.025,k-1)
    grand=np.concatenate(groups).mean()
    ms_between=sum(len(group)*(group.mean()-grand)**2 for group in groups)/(k-1)
    ms_within=sum(float(((group-group.mean())**2).sum()) for group in groups)/(n-k) if n>k else None
    n0=(n-sum(len(group)**2 for group in groups)/n)/(k-1)
    denominator=ms_between+(n0-1)*ms_within if ms_within is not None else 0
    icc=float((ms_between-ms_within)/denominator) if denominator else None
    def required(v):
        if v<=0:
            return None  # zero development variance is not population certainty
        for count in range(5,100001):
            if float(t.ppf(.975,count-1))*sqrt(v/count)<=half_width:
                return count
        return 'EXCEEDS_100000'
    return {'scope':'DEVELOPMENT_PLANNING_ONLY','cluster_unit':'original source-text speaker',
            'target_effect':target_effect,'target_ci_half_width':half_width,'clusters_observed':k,'recordings_observed':n,
            'equal_cluster_mean':float(means.mean()),'cluster_mean_variance':variance,
            'variance_interval_normal_iid_clusters_assumption':[float(lower),float(upper)],
            'observed_t95_half_width':critical*sqrt(variance/k),'icc_method_of_moments':icc,
            'design_effect_using_nonnegative_icc':float(1+(n/k-1)*max(0,icc)) if icc is not None else None,
            'required_clusters_at_pilot_variance':required(variance),'required_clusters_at_variance_upper_bound':required(upper),
            'status':'INSUFFICIENT_INDEPENDENT_GROUPS' if k<20 else 'DEVELOPMENT_VARIANCE_ESTIMATE',
            'limitations':'Only pilot source-text clusters; t/chi-square/ICC assumptions unverified; planning, not power or an independence certificate; zero variance cannot justify zero samples'}
