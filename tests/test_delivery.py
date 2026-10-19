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
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'reports/exercises/generated/manifest.json').read_text())
    expected = 'Báo cáo chưa có kết quả thực nghiệm' if manifest['train_runs'] == 0 else f"{manifest['train_runs']} lượt huấn luyện"
    assert expected in text
    assert (target / 'reports/exercises.pdf').is_file()
    assert (target / 'results.json').is_file()
    assert 'href="favicon.svg"' in text
    assert (target / 'favicon.svg').is_file()


def test_publish_plan_is_explicit_and_never_force_pushes():
    from scripts.publish_github import create_command
    command = create_command('dathuynh1108', 'public')
    assert command[:4] == ['gh', 'repo', 'create', 'dathuynh1108/CO5085-deep-learning']
    assert '--public' in command and '--push' in command
    assert '--force' not in command
    with pytest.raises(ValueError):
        create_command('someone-else', 'public')


def test_site_builder_never_writes_into_report_sources(tmp_path):
    from scripts.build_site import build_site, ROOT
    with pytest.raises(ValueError):
        build_site(ROOT / 'reports/exercises')


def test_src_packages_and_run_guards_use_current_paths():
    from common.config import ROOT
    from A1.engine import source_identity
    from common.engine import source_fingerprint
    assert (ROOT / 'configs/experiments.json').is_file()
    assert all(p.startswith('src/') for p in source_identity())
    assert len(source_fingerprint()) == 64


def test_archive_mode_is_explicit_and_cannot_replace_current_training_source():
    from common.archive import a1_source
    from A1.export_report import read_run, expected_source
    from A1.protocol import ROOT, load_protocol
    protocol=load_protocol()
    experiment=protocol['experiments'][0]
    run=ROOT / 'reports/assignment1/generated/evidence' / experiment['id'] / 'seed_42'
    with pytest.raises(ValueError, match='source'):
        read_run(run, experiment, 42, protocol)
    historical=a1_source()
    assert historical != expected_source()
    row,status=read_run(run, experiment, 42, protocol, source_files=historical)
    assert status == 'complete' and row['test_accuracy'] is not None


def test_site_does_not_invent_sd_for_single_seed():
    from scripts.build_site import accuracy_text
    assert accuracy_text(.9, None) == '90.00% · SD chưa xác định'
    assert accuracy_text(None, None) == 'Chưa đo'
    assert accuracy_text(.9, .01) == '90.00 ± 1.00%'
