import json
from unittest.mock import patch

from scripts.jaes_v2 import medium_progress as monitor


IDENTITY = 'p10_medium_erm_s9207_v2'
REGISTRY = dict(backend='FULL_SHAPE_RECOMPUTE', maximum_updates=840,
                paired_runs=[dict(experiment_id=IDENTITY)])
CONTRACT = dict(config=dict(architecture='MobileNetV3Small-LSTM128', warmup_epochs=1,
                            finetune_epochs=20))


def test_progress_states():
    assert 'Waiting' in monitor.render(IDENTITY, REGISTRY, None)
    active = dict(status='RUNNING', step=39, epoch=0)
    assert '39/840' in monitor.render(IDENTITY, REGISTRY, active, CONTRACT)
    active.update(step=40, epoch=1, validation=dict(tier='A', clean=dict(roc_auc=.75, fp=7, n_REAL=112)))
    assert 'Epoch 1/21' in monitor.render(IDENTITY, REGISTRY, active, CONTRACT)
    assert 'Clean ROC-AUC 0.75' in monitor.render(IDENTITY, REGISTRY, active, CONTRACT)
    assert 'N/A' in monitor.render(IDENTITY, REGISTRY, active, CONTRACT)
    assert 'process exit' in monitor.render(IDENTITY, REGISTRY, active, CONTRACT, stale=True)
    active.update(status='WARNING', warning='resource pressure')
    assert 'Warning resource pressure' in monitor.render(IDENTITY, REGISTRY, active)
    active.update(status='PAUSE_AT_CHECKPOINT', latest=dict(checkpoint='u000040_e001_c000'))
    assert 'u000040_e001_c000' in monitor.render(IDENTITY, REGISTRY, active)
    active.update(status='SCIENTIFIC_COMPLETE', step=840)
    assert '840/840' in monitor.render(IDENTITY, REGISTRY, active)


def test_missing_malformed_and_interrupted(tmp_path, capsys):
    path = tmp_path/'status.json'
    assert monitor.read_json(path) is None
    path.write_text('{')
    assert monitor.read_json(path) is None
    path.write_text(json.dumps(dict(step=40)))
    assert monitor.read_json(path) == {'step': 40}
    with (patch.object(monitor, 'read_json', side_effect=[REGISTRY, None, None]),
          patch.object(monitor.time, 'sleep', side_effect=KeyboardInterrupt),
          patch.object(monitor.sys, 'argv', ['progress','--experiment',IDENTITY])):
        monitor.main()
    assert 'Monitor stopped.' in capsys.readouterr().out


def test_recorded_metrics_and_resume_event(tmp_path):
    events = tmp_path/'resume_events'
    events.mkdir()
    (events/'u000040.json').write_text('{"step":40}', encoding='utf-8')
    (events/'u000080.json').write_text('{"step":80}', encoding='utf-8')
    (events/'u000120.json').write_text('{', encoding='utf-8')
    status = dict(status='RUNNING', step=280, epoch=7, remaining_train_seconds=5465,
                  last_update=dict(loss=.125, lr=.0001, norm=.2),
                  validation=dict(step=280, clean=dict(nll=.35)))
    screen = monitor.render(IDENTITY, REGISTRY, status, CONTRACT, elapsed=2718,
                            resume=monitor.latest_resume(tmp_path))
    for expected in ('280/840 — 33.3%', '00:45:18', 'Optimizer ETA 01:31:05',
                     'Clean BCE 0.35', 'Recorded at update 80', 'Training accuracy and AUC N/A'):
        assert expected in screen
    assert 'differs from the registered budget' in monitor.render(
        IDENTITY, REGISTRY, status, dict(config=dict(maximum_updates=841)))
