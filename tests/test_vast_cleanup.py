"""Destructive cleanup must fail closed when the backup or identity is wrong."""
import hashlib
import json
from pathlib import Path
import subprocess
import pytest
from scripts import vast_finalize, watch_vast_training


@pytest.mark.parametrize('problem', ['checksum', 'endpoint'])
def test_cleanup_never_destroys_unverified_backup_or_wrong_instance(tmp_path, monkeypatch, problem):
    monkeypatch.setattr(vast_finalize, 'ROOT', tmp_path)
    monkeypatch.setattr(vast_finalize, 'validate_completion', lambda root: {'runs': 60, 'runs/assignment1': 24})
    fixture = tmp_path / 'runs/fixture.pt'
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(b'checkpoint fixture, not a trained model')
    expected_hash = hashlib.sha256(fixture.read_bytes()).hexdigest() if problem == 'endpoint' else 'wrong'
    called = []

    def run(command, **kwargs):
        called.append(command)
        assert command[:3] != ['vastai', 'destroy', 'instance']
        return None

    def output(command, **kwargs):
        called.append(command)
        if command[0] == 'ssh':
            return json.dumps({'runs/fixture.pt': expected_hash})
        return json.dumps({'instances': [{'id': 123, 'public_ipaddr': 'different-host', 'ports': {}}]})

    monkeypatch.setattr(vast_finalize.subprocess, 'run', run)
    monkeypatch.setattr(vast_finalize.subprocess, 'check_output', output)
    monkeypatch.setattr('sys.argv', ['cleanup', '--instance', '123', '--host', 'expected-host',
                                    '--port', '12345', '--identity', str(tmp_path / 'key'), '--delete'])
    with pytest.raises(ValueError, match='checksum|endpoint'):
        vast_finalize.main()
    assert not any(c[:3] == ['vastai', 'destroy', 'instance'] for c in called)
    manifest = tmp_path / 'docs/training-evidence/backup-manifest.json'
    if manifest.exists():
        assert not json.loads(manifest.read_text())['instance_deleted']


def test_failed_gpu_job_is_preserved_for_diagnosis(tmp_path, monkeypatch):
    monkeypatch.setattr(watch_vast_training, 'ROOT', tmp_path)
    snapshot = {'complete': False, 'exit': 1, 'exercise_trained': 12, 'a1_records': 0}
    monkeypatch.setattr(watch_vast_training.subprocess, 'check_output', lambda *a, **k: json.dumps(snapshot))
    monkeypatch.setattr(watch_vast_training.subprocess, 'run', lambda *a, **k: pytest.fail('No finalization on failed job'))
    monkeypatch.setattr('sys.argv', ['watch', '--instance', '123', '--host', 'host', '--port', '12345',
                                    '--identity', str(tmp_path / 'key')])
    assert watch_vast_training.main() == 1
    assert json.loads((tmp_path / 'artifacts/vast-watch-status.json').read_text())['state'] == 'attention_required'


@pytest.mark.parametrize('exits, expected_attempts, fails', [
    ([255, 0], 2, False),
    ([255, 255, 255], 3, True),
    ([23], 1, True),
])
def test_transfer_retries_only_ssh_disconnects(monkeypatch, exits, expected_attempts, fails):
    attempts = []

    def run(command, **kwargs):
        attempts.append(command)
        code = exits[len(attempts) - 1]
        if code:
            raise subprocess.CalledProcessError(code, command)

    monkeypatch.setattr(vast_finalize.subprocess, 'run', run)
    monkeypatch.setattr(vast_finalize.time, 'sleep', lambda _: None)
    if fails:
        with pytest.raises(subprocess.CalledProcessError):
            vast_finalize.copy_with_retry(['rsync', 'fixture'])
    else:
        vast_finalize.copy_with_retry(['rsync', 'fixture'])
    assert len(attempts) == expected_attempts
