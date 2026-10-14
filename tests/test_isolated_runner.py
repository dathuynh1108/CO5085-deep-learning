"""Operational isolation checks; no datasets, training, or GPU calls."""
import subprocess
import sys

import pytest

from common.config import Experiment
from scripts import run_exercises_isolated as runner


@pytest.fixture
def launches(monkeypatch):
    configs = [Experiment(id='first', exercise='E1', model='cnn', label='First'),
               Experiment(id='second', exercise='E3', model='lstm', label='Second')]
    monkeypatch.setattr(runner, 'load_experiments', lambda: configs)
    calls = []

    def launch(command, *, check):
        assert check is True
        calls.append(command)

    monkeypatch.setattr(runner.subprocess, 'run', launch)
    return calls


def test_train_isolates_each_seed_and_keeps_resume_guards(tmp_path, launches):
    output = tmp_path / 'runs with spaces'
    checkpoint = output / 'first' / 'seed_42' / 'last.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    runner.main(['--group', 'all', '--seeds', '42', '43', '--device', 'cuda',
                 '--num-workers', '8', '--output-root', str(output), '--resume'])
    assert len(launches) == 4
    assert [call[2] for call in launches] == ['E1.train', 'E1.train', 'E3.train', 'E3.train']
    assert [call[call.index('--seed') + 1] for call in launches] == ['42', '43', '42', '43']
    assert [('--resume' in call) for call in launches] == [True, False, False, False]
    for call in launches:
        assert call[:2] == [sys.executable, '-m']
        assert call[call.index('--output-root') + 1] == str(output)
        assert call[-4:] == ['--device', 'cuda', '--num-workers', '8']
        assert '--epochs' not in call and '--batch-size' not in call


def test_test_requires_permission_before_launching_and_never_forces(launches):
    with pytest.raises(SystemExit):
        runner.main(['--phase', 'test'])
    assert launches == []
    runner.main(['--phase', 'test', '--confirm-test', '--group', 'all', '--seeds', '42', '43'])
    assert len(launches) == 4
    for call in launches:
        assert call[:3] == [sys.executable, '-m', 'scripts.evaluate']
        assert '--confirm-test' in call and '--force' not in call
        assert call[call.index('--split') + 1] == 'test'


def test_failed_child_stops_remaining_experiments(monkeypatch, launches):
    def fail(command, *, check):
        launches.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(runner.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        runner.main(['--group', 'all', '--seeds', '42', '43'])
    assert len(launches) == 1


def test_list_and_repeated_seeds_do_not_launch(launches):
    runner.main(['--list', '--group', 'all'])
    with pytest.raises(SystemExit):
        runner.main(['--seeds', '42', '42'])
    assert launches == []
