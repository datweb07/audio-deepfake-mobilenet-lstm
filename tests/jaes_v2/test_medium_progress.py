import hashlib
import io
import json
import time

import pytest

from scripts.jaes_v2 import medium_progress as monitor


CONFIG = dict(architecture='MobileNetV3Small-LSTM128', warmup_epochs=1, finetune_epochs=20)
IDENTITY = 'p10_medium_channel_s9307_v2'


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


def state(step, epoch, last_epoch, **kwargs):
    return dict(status='RUNNING', step=step, epoch=epoch,
                last_update=dict(step=step, epoch=last_epoch, loss=.625, lr=.0001, norm=.25),
                remaining_train_seconds=5465, **kwargs)


def validation(step, epoch):
    return dict(step=step, epoch=epoch, tier='A',
                clean=dict(roc_auc=.75, nll=.4, fp=7, n_REAL=112, fn=8, n_FAKE=112))


def snapshots(tmp_path, monkeypatch, entries):
    values = iter(entries)
    log = io.StringIO()
    def advance():
        item = next(values, None)
        if item is None:
            raise KeyboardInterrupt
        write(tmp_path/'status.json', item)
    monkeypatch.setattr(monitor.time, 'sleep', lambda _: advance())
    advance()
    with pytest.raises(KeyboardInterrupt):
        monitor.monitor(tmp_path, CONFIG, 840, monitor.Terminal(log, live=False))
    return log.getvalue()


def test_synthetic_epochs_no_duplicate_validation(tmp_path, monkeypatch):
    one = state(40,1,0,validation=validation(40,1),best=dict(tag='u000040_e001_c000'))
    two = state(41,1,1,validation=one['validation'],best=one['best'])
    three = state(80,2,1,validation=validation(80,2),best=one['best'])
    four = state(81,2,2,validation=three['validation'],best=one['best'])
    out = snapshots(tmp_path,monkeypatch,[state(12,0,0),one,one,two,three,four])
    assert out.count('Epoch 1/21') == out.count('Epoch 2/21') == out.count('Epoch 3/21') == 1
    assert out.index('Validation (epoch 1)') < out.index('Epoch 2/21')
    assert out.count('Validation (epoch') == 2 and out.count('40/840') == 1
    assert 'Best checkpoint updated at step 40' in out and 'Best checkpoint unchanged.' in out
    assert 'val_auc: 0.7500 - val_bce: 0.4000 - REAL_FA: 7/112 - FAKE_miss: 8/112' in out
    assert 'loss: 0.6250 - lr: 1.00e-04 - grad_norm: 0.2500' in out
    assert 'train_ETA: 01:31:05' in out and 'monitor_elapsed:' in out
    assert '\r' not in out and '\x1b' not in out
    assert 'accuracy' not in out
    print('SIMULATED FIXTURE, NOT SCIENTIFIC RESULTS\n'+out, end='')


@pytest.mark.parametrize('value',[None, {}, {'status':[]}, {'status':'BROKEN','step':0,'epoch':0},
                                 {'status':'RUNNING','step':841,'epoch':0},
                                 {'status':'RUNNING','step':True,'epoch':0}])
def test_invalid_status(value):
    assert not monitor.valid_status(value,840,21)


def test_missing_partial_and_interrupt(tmp_path, monkeypatch, capsys):
    status=tmp_path/'scientific/status.json'
    assert monitor.read_json(status) is None
    status.parent.mkdir()
    status.write_text('{')
    assert monitor.read_json(status) is None
    monkeypatch.setattr(monitor,'BASE',tmp_path)
    monkeypatch.setattr(monitor.sys,'argv',['monitor','--experiment',IDENTITY])
    def interrupt(_):
        raise KeyboardInterrupt
    monkeypatch.setattr(monitor.time,'sleep',interrupt)
    monitor.main()
    assert 'Waiting for readable status.json' in capsys.readouterr().out
    assert status.read_text()=='{'


def test_duration_progress_and_no_early_completion():
    assert monitor.duration(5465)=='01:31:05'
    assert all(monitor.duration(v)=='N/A' for v in (None,-1,float('nan'),'5'))
    assert '33.3%' in monitor.progress(state(280,7,6),840)[0]
    assert '<100%' in monitor.progress(state(840,21,20),840)[0]
    assert '99.9%' in monitor.progress(state(49999,21,20),50000)[0]
    assert 'train_ETA: N/A' in monitor.progress(dict(status='RUNNING',step=1),840)[1]
    assert monitor.validation_line({'epoch':1,'clean':{'nll':'invalid'}}).endswith('N/A')


def test_ascii_utf8_resizing_and_live_output(monkeypatch):
    width=120
    monkeypatch.setattr(monitor.shutil,'get_terminal_size',lambda _:type('Size',(),{'columns':width})())
    log=io.StringIO()
    term=monitor.Terminal(log,live=True)
    s=state(280,7,6)
    assert '█' in monitor.progress(s,840)[0] and '#' in monitor.progress(s,840,unicode=False)[0]
    assert not monitor.supports_unicode(type('Stream',(),{'encoding':'cp1252'})())
    assert monitor.supports_unicode(type('Stream',(),{'encoding':'utf-8'})())
    term.update(monitor.progress(s,840,width=width))
    width=80
    term.update(monitor.progress(s,840,width=width))
    assert len(log.getvalue().rsplit('\r',1)[1]) <= 79
    assert 'train_ETA:' in log.getvalue().rsplit('\r',1)[1]
    assert '\x1b' not in log.getvalue()
    term.message('Checkpoint saved.')
    assert log.getvalue().endswith('\nCheckpoint saved.\n')


@pytest.mark.parametrize('stage',['PAUSE_AT_CHECKPOINT','STOP_FOR_RESOURCE_SAFETY',
                                  'STOP_FOR_NUMERICAL_FAILURE','SCIENTIFIC_NO_GO'])
def test_terminal_stops_and_warnings(tmp_path,stage):
    s=state(40,1,0,latest=dict(checkpoint='u000040_e001_c000'),
            warning='RESOURCE_FIXTURE',resources=dict(ram_available=2**30))
    write(tmp_path/'status.json',dict(s,status=stage))
    log=io.StringIO()
    monitor.monitor(tmp_path,CONFIG,840,monitor.Terminal(log,live=False))
    out=log.getvalue()
    assert stage in out and 'u000040_e001_c000' in out
    assert 'Training completed' not in out
    assert ('RAM 1.00 GiB' in out) == (stage=='STOP_FOR_RESOURCE_SAFETY')


def test_resume_and_stale_are_read_only(tmp_path, monkeypatch):
    write(tmp_path/'resume_events/u000080.json',dict(step=80))
    (tmp_path/'resume_events/u000120.json').write_text('{')
    write(tmp_path/'status.json',state(81,2,2))
    before={p:hashlib.sha256(p.read_bytes()).digest() for p in tmp_path.rglob('*.json')}
    now=time.time()
    monkeypatch.setattr(monitor.time,'time',lambda:now+121)
    polls=iter([0,0,0])
    def wait(_):
        if next(polls,None) is None:
            raise KeyboardInterrupt
    monkeypatch.setattr(monitor.time,'sleep',wait)
    log=io.StringIO()
    with pytest.raises(KeyboardInterrupt):
        monitor.monitor(tmp_path,CONFIG,840,monitor.Terminal(log,live=False))
    out=log.getvalue()
    assert out.count('Resume recorded at update 80')==1
    assert out.count('No recent status write')==1
    assert 'process exit' in out and 'Epoch 3/21' in out
    assert before=={p:hashlib.sha256(p.read_bytes()).digest() for p in tmp_path.rglob('*.json')}


def test_completion_waits_for_seal(tmp_path):
    s=state(840,21,20,best=dict(tag='u000720_e018_c000'))
    write(tmp_path/'status.json',dict(s,status='SCIENTIFIC_COMPLETE'))
    write(tmp_path/'result.json',{'state':{'step':840}})
    log=io.StringIO()
    monitor.monitor(tmp_path,CONFIG,840,monitor.Terminal(log,live=False),once=True)
    assert 'Waiting for the completion seal' in log.getvalue() and '<100%' in log.getvalue()
    write(tmp_path/'seal.json',{'output_hashes':{'result.json':'fixture','status.json':'fixture'}})
    log=io.StringIO()
    monitor.monitor(tmp_path,CONFIG,840,monitor.Terminal(log,live=False),once=True)
    assert '100.0%' in log.getvalue() and 'Training completed.' in log.getvalue()
    assert 'Best checkpoint: u000720_e018_c000' in log.getvalue()
