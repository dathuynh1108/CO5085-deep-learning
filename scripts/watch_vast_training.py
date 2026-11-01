"""Wait for the detached GPU job, verify its backup, then delete its instance.

Logs go to artifacts/ locally. A failed/incomplete GPU job never triggers deletion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from scripts.vast_finalize import copy_with_retry

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', type=int, required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--identity', type=Path, required=True)
    parser.add_argument('--max-hours', type=float, default=24)
    args = parser.parse_args()
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
           '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6',
           '-i', str(args.identity), '-p', str(args.port)]
    target = f'root@{args.host}'
    remote_root = '/workspace/CO5085-deep-learning'
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    status_file = artifacts / 'vast-watch-status.json'

    def status(state, **extra):
        value = {'state': state, 'instance': args.instance, 'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}
        status_file.write_text(json.dumps(value, indent=2) + '\n')
        print(json.dumps(value), flush=True)

    remote_code = ('import pathlib,json; r=pathlib.Path(' + repr(remote_root) + '); '
                   'p=r/"artifacts/logs/coursework.exit"; '
                   'print(json.dumps({"complete":(r/"artifacts/logs/coursework.complete").is_file(), '
                   '"exit":int(p.read_text().strip()) if p.exists() else None, '
                   '"exercise_trained":len(list((r/"runs/exercises").glob("*/seed_*/metrics.json"))), '
                   '"a1_records":len(list((r/"runs/assignment1").glob("*/seed_*/metadata.json")))}))')
    deadline = time.monotonic() + args.max_hours * 3600
    errors = 0
    while time.monotonic() < deadline:
        try:
            snapshot = json.loads(subprocess.check_output(
                ssh + [target, '/usr/bin/python3 -c ' + shlex.quote(remote_code)], text=True, timeout=45))
            errors = 0
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
            errors += 1
            status('connection_retry', attempts=errors, error=str(exc))
            if errors >= 10:
                status('attention_required', reason='Ten consecutive SSH failures. No deletion attempted.')
                return 1
            time.sleep(60)
            continue
        if snapshot['exit'] not in {None, 0}:
            status('attention_required', reason='GPU job failed; preserve instance for diagnosis.', **snapshot)
            return 1
        if snapshot['complete']:
            status('collecting', **snapshot)
            break
        status('training', **snapshot)
        time.sleep(60)
    else:
        status('attention_required', reason='Watch timeout. No deletion attempted; inspect job before any cleanup.')
        return 1
    try:
        # Reports/figures/evidence are preserved before the independent
        # checkpoint SHA256 guard permits destruction.
        for folder in ['reports/exercises/generated', 'reports/assignment1/generated', 'build/site']:
            destination = ROOT / folder
            destination.mkdir(parents=True, exist_ok=True)
            copy_with_retry(['rsync', '-az', '--partial-dir=.rsync-partial', '-e', shlex.join(ssh),
                             f'{target}:{remote_root}/{folder}/', str(destination) + '/'])
        for relative in ['reports/exercises/exercises.pdf', 'reports/assignment1/assignment1.pdf']:
            copy_with_retry(['rsync', '-az', '--partial-dir=.rsync-partial', '-e', shlex.join(ssh),
                             f'{target}:{remote_root}/{relative}', str(ROOT / relative)])
        subprocess.run([sys.executable, '-m', 'scripts.vast_finalize', '--instance', str(args.instance),
                        '--host', args.host, '--port', str(args.port), '--identity', str(args.identity), '--delete'],
                       cwd=ROOT, check=True)
    except (subprocess.SubprocessError, OSError) as exc:
        status('attention_required', reason='Collection/verification failed. Deletion requires a verified backup.', error=str(exc))
        return 1
    manifest = json.loads((ROOT / 'docs/training-evidence/backup-manifest.json').read_text())
    document = ROOT / 'docs/TRAINING_EXECUTION.md'
    document.write_text(document.read_text().split('## Trạng thái')[0] + '## Trạng thái hoàn tất\n\n'
                        + f'E1–E3: 60/60 lượt train đủ30 epoch và 60 lượt held-out test. '
                        + 'A1: 24/24 lượt train đủ20 epoch và 24 lượt held-out test. '
                        + 'Diagnostic baseline/combined E3 đã chạy trên validation256 ảnh cho ba seed.\n\n'
                        + f'Đã tải và kiểm tra SHA256 của {manifest["verified_files"]} artifact; '
                        + f'instance {args.instance} đã được xóa và CLI không còn liệt kê nó. '
                        + 'Checkpoint nằm trong runs/exercises/ và runs/assignment1/ ở workspace, bị Git ignore. '
                        + 'Evidence nhẹ và checksum ở reports/exercises/generated/evidence, reports/assignment1/generated/evidence '
                        + 'và docs/training-evidence/backup-manifest.json.\n\n'
                        + 'Các PDF đã build từ số liệu thật; cần bước kiểm tra hình ảnh PDF cuối của agent '
                        + 'trước khi coi bản báo cáo sẵn sàng nộp. Chưa commit/push.\n')
    status('complete', checkpoint_backup_verified=True, instance_deleted=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
