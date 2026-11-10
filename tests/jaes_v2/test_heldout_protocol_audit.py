from io import BytesIO
import json
from unittest.mock import patch
from zipfile import ZipFile

import pytest

from scripts.jaes_v2 import heldout_protocol_audit as audit
from src.lava.jaes_v2.evidence import file_hash

ROWS = b'LA_0001 LA_E_1 - - bonafide\nLA_0002 LA_E_2 - - bonafide\nLA_0001 LA_E_3 - A07 spoof\n'


def fixture(protocol=ROWS, omit=(), duplicate_bytes=False):
    stream = BytesIO()
    with ZipFile(stream, 'w') as archive:
        members = {audit.CM:protocol, audit.LICENSE:b'fixture license'}
        members.update({f'{audit.EVAL}LA_E_{i}.flac':bytes([i if not duplicate_bytes else 1])
                        for i in range(1,5)})
        for name, data in members.items():
            if name not in omit:
                archive.writestr(name, data)
    return ZipFile(stream)


def inspect(archive, metadata=(), count=2):
    with archive:
        return audit.inspect_archive(archive, metadata, dict(REAL=count,per_attack=count), 91109, {0:2,1:1})


def test_inventory_sampling_and_fallback():
    one, two = inspect(fixture()), inspect(fixture())
    assert one == two
    assert one['zip_without_cm_trial'] == 1
    assert one['support']['A08']['selected'] == 0
    assert {r['speaker_id'] for r in one['rows'] if r['label'] == 0} == {'LA_0001','LA_0002'}
    fallback = inspect(fixture(), count=1)
    assert fallback['support']['REAL']['selected'] == 1
    assert fallback['selected_rows_sha256'] != one['selected_rows_sha256']


@pytest.mark.parametrize('omit',[audit.CM, audit.LICENSE, f'{audit.EVAL}LA_E_2.flac'])
def test_missing_archive_members(omit):
    with pytest.raises(ValueError,match='Missing'):
        inspect(fixture(omit=[omit]))


@pytest.mark.parametrize('protocol',[ROWS+ROWS, ROWS.replace(b'A07',b'A02'), ROWS+b'bad line',
                                    ROWS.replace(b'LA_0002',b'bad')])
def test_corrupt_protocol(protocol):
    with pytest.raises(ValueError):
        inspect(fixture(protocol))


def test_overlap_checks():
    for row in (dict(sample_id='LA_E_1',speaker_id='none',checksum='none'),
                dict(sample_id='none',speaker_id='LA_0002',checksum='none')):
        with pytest.raises(ValueError,match='overlapping'):
            inspect(fixture(), [row])
    with pytest.raises(ValueError,match='audio bytes'):
        inspect(fixture(duplicate_bytes=True))


def test_separate_approval_and_safe_stop(tmp_path):
    pre = tmp_path/audit.PRELOCK
    pre.parent.mkdir(parents=True)
    approval = tmp_path/'author.json'
    output = tmp_path/audit.OUT/'proposed.json'
    def put(path,value):
        path.write_text(json.dumps(value),encoding='utf-8')
    put(pre,dict(authorized=False,execution_status='HELDOUT_NOT_AUTHORIZED',files={},
                 seeds=[9107,9207],conditions=['clean'],ffmpeg_sha256='x',benchmark_sha256='y'))
    for name in (audit.AUDITOR,audit.RUNNER,audit.PROTOCOL):
        p=tmp_path/name
        p.parent.mkdir(parents=True,exist_ok=True)
        put(p,dict(heldout=dict(fallback=dict(REAL=1,per_attack=1),sampling_seed=91109)))
    foundation=tmp_path/audit.FOUNDATION
    foundation.mkdir(parents=True)
    for role in ('train','development','calibration'):
        (foundation/f'{role}_manifest.csv').write_text('sample_id,speaker_id,checksum\n')
    common=dict(author='fixture',approved_at='fixture',approval_reference='fixture')
    put(approval,dict(common,scope='AUTHOR_APPROVAL_HELDOUT_SCORING',prelock_sha256=file_hash(pre)))
    with patch.object(audit,'ZipFile') as archive, pytest.raises(ValueError,match='PROTOCOL_READ'):
        audit.run(tmp_path,approval,output)
    archive.assert_not_called()
    auth=dict(common,scope='AUTHOR_APPROVAL_PROTOCOL_READ',prelock_sha256=file_hash(pre),
              allow_selected_audio_byte_hashes=True,subset='fallback',resource_subset_reason='fixture')
    original=audit.read(pre)
    put(pre,dict(original,files={audit.PROTOCOL.as_posix():'invalid digest'}))
    put(approval,dict(auth,prelock_sha256=file_hash(pre)))
    with patch.object(audit,'ZipFile') as archive, pytest.raises(ValueError,match='Prelock input mismatch'):
        audit.run(tmp_path,approval,output)
    archive.assert_not_called()
    put(pre,original)
    put(approval,auth)
    with (patch.object(audit,'ZipFile',return_value=fixture()),
          patch.object(audit,'OFFICIAL_COUNTS',{0:2,1:1})):
        audit.run(tmp_path,approval,output)
    proposed=audit.read(output)
    assert proposed['authorized'] is False and proposed['subset']=='fallback'
    with pytest.raises(ValueError,match='SCORING'):
        audit.scoring_lock(output,approval)
    result=proposed['evaluation_manifest']
    put(approval,dict(common,scope='AUTHOR_APPROVAL_HELDOUT_SCORING',
                     proposed_lock_sha256=file_hash(output),
                     selected_rows_sha256=result['selected_rows_sha256'],
                     license_sha256=result['license']['sha256']))
    assert audit.scoring_lock(output,approval)['authorized'] is True
    put(approval,dict(audit.read(approval),selected_rows_sha256='bad'))
    with pytest.raises(ValueError,match='does not bind'):
        audit.scoring_lock(output,approval)
