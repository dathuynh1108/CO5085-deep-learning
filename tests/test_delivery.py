"""Packaging checks: no network calls or model training."""
from pathlib import Path
import json
import pytest


def test_static_site_copies_pdf_and_reports_actual_empty_status(tmp_path):
    from scripts.build_site import build_site
    target = tmp_path / 'site'
    build_site(target)
    text = (target / 'index.html').read_text()
    assert 'Huỳnh Thành Đạt' in text and '2570161' in text
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'reports/generated/manifest.json').read_text())
    expected = 'Chưa huấn luyện' if manifest['train_runs'] == 0 else f"{manifest['train_runs']} lượt train"
    assert expected in text
    assert (target / 'reports/exercises.pdf').is_file()
    assert (target / 'results.json').is_file()


def test_publish_plan_is_explicit_and_never_force_pushes():
    from scripts.publish_github import create_command
    command = create_command('dathuynh1108', 'public')
    assert command[:4] == ['gh', 'repo', 'create', 'dathuynh1108/CO5085-deep-learning']
    assert '--public' in command and '--push' in command
    assert '--force' not in command
    with pytest.raises(ValueError):
        create_command('someone-else', 'public')
