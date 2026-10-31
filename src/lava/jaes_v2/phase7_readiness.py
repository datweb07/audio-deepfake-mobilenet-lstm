"""Fail-closed external/calibration preparation. Does not open reserved data."""
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class ConfirmationContract:
    status:str='NOT_LOCKED'
    hypothesis:str=''
    official_source:str=''
    version:str=''
    license_verified:bool=False
    storage_verified:bool=False
    overlap_verified:bool=False
    source_speaker_attack_metadata_verified:bool=False
    model_checkpoint_sha256:str=''
    training_complete:bool=False
    hyperparameters_frozen:bool=False
    operating_policy_frozen:bool=False
    primary_comparisons_registered:bool=False
    previous_development_disclosed:bool=True
    calibration_policy:str='RESERVED_UNUSED'
    independent_confirmation:bool=False

def readiness(contract):
    checks={k:bool(getattr(contract,k)) for k in ('hypothesis','official_source','version','license_verified','storage_verified','overlap_verified','source_speaker_attack_metadata_verified','training_complete','hyperparameters_frozen','operating_policy_frozen','primary_comparisons_registered','previous_development_disclosed')}
    checks['checkpoint_sha256']=bool(re.fullmatch('[a-f0-9]{64}',contract.model_checkpoint_sha256))
    checks['explicitly_locked']=contract.status=='LOCKED'
    return dict(status='LOCKED' if all(checks.values()) else 'NOT_LOCKED',checks=checks,
        allowed_data_roles=[] if not all(checks.values()) else ['DECLARED_EXTERNAL_ONLY'],calibration_accessed=False,evaluation_accessed=False)

def require_locked(contract):
    if readiness(contract)['status']!='LOCKED': raise ValueError('EXTERNAL_VALIDATION_NOT_LOCKED; labels/audio remain closed')

@dataclass(frozen=True)
class CalibrationContract:
    status:str='NOT_LOCKED'
    fitting_partition:str='reserved_calibration'
    fitting_manifest_sha256:str=''
    independent_from_selection:bool=False
    selected_checkpoint_sha256:str=''
    method:str='TEMPERATURE_SCALING'
    threshold_selection_rule:str='PREDECLARED_REAL_FALSE_ALARM_TARGET'
    risk_guarantee:str='NONE_UNDER_ARBITRARY_SHIFT'
    def validate_for_fit(self):
        if self.status!='LOCKED' or self.fitting_partition!='reserved_calibration' or not self.independent_from_selection or self.method!='TEMPERATURE_SCALING' or not all(re.fullmatch('[a-f0-9]{64}',s) for s in (self.fitting_manifest_sha256,self.selected_checkpoint_sha256)): raise ValueError('Calibration fit NOT_LOCKED')

def temperature_transform(probabilities,temperature):
    """Apply a supplied temperature only; never estimates it from reserved labels."""
    import numpy as np
    p=np.asarray(probabilities,dtype='float64')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not np.isfinite(temperature) or temperature<=0: raise ValueError('Invalid temperature/score')
    p=np.clip(p,1e-7,1-1e-7); z=(np.log(p)-np.log1p(-p))/temperature
    return np.exp(-np.logaddexp(0,-z))
