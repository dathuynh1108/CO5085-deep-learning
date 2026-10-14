"""Resume the official CIFAR-100 archive; expose it only after full MD5 verification."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request


def digest(path: Path, algorithm: str) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def fetch_range(url: str, start: int, end: int, total: int | None = None) -> tuple[bytes, int]:
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(request, timeout=30) as response:
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
        if response.status != 206 or not match:
            raise ValueError('Official server did not honor the requested byte range.')
        first, last, length = map(int, match.groups())
        if (first, last) != (start, end) or (total is not None and length != total):
            raise ValueError('Official server returned a different archive range or size.')
        data = response.read(end - start + 1)
    if len(data) != end - start + 1:
        raise OSError('Incomplete archive range; no partial chunk was appended.')
    return data, length


def download_archive(url: str, destination: Path, expected_md5: str, progress: Path,
                     chunk_bytes: int = 512 * 1024, attempts: int = 3) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    progress.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(destination.name + '.download.lock')
    # A second downloader must not append to the same archive concurrently.
    with lock.open('x') as stream:
        stream.write(str(os.getpid()) + '\n')
    partial = destination.with_name(destination.name + '.partial')
    audit = {'source': url, 'expected_md5': expected_md5, 'chunk_bytes': chunk_bytes,
             'started_utc': datetime.now(timezone.utc).isoformat(), 'state': 'downloading'}

    def save():
        temporary = progress.with_name(progress.name + '.tmp')
        temporary.write_text(json.dumps(audit, indent=2) + '\n')
        temporary.replace(progress)

    try:
        if destination.exists() and digest(destination, 'md5') == expected_md5:
            audit.update(state='verified', cached=True, bytes=destination.stat().st_size,
                         actual_md5=expected_md5, sha256=digest(destination, 'sha256'))
            return audit
        _, total = fetch_range(url, 0, 0)
        audit['archive_bytes'] = total
        if destination.exists():
            if partial.exists():
                raise FileExistsError('Both unverified archive and partial exist; inspect before choosing one.')
            audit['initial_unverified_prefix_sha256'] = digest(destination, 'sha256')
            destination.rename(partial)
        partial.touch(exist_ok=True)
        if partial.stat().st_size > total:
            raise ValueError('Unverified partial is larger than the official archive.')
        audit['initial_bytes'] = partial.stat().st_size
        save()
        deadline = time.monotonic() + 2 * 3600
        while partial.stat().st_size < total:
            if time.monotonic() > deadline:
                raise TimeoutError('Two-hour archive download limit reached; partial retained.')
            start = partial.stat().st_size
            end = min(start + chunk_bytes, total) - 1
            for attempt in range(attempts):
                try:
                    data, _ = fetch_range(url, start, end, total)
                    break
                except (OSError, urllib.error.URLError, http.client.IncompleteRead):
                    if attempt + 1 == attempts:
                        raise
                    time.sleep(2)
            with partial.open('ab') as stream:
                stream.write(data)
                stream.flush()
            audit.update(bytes=partial.stat().st_size, updated_utc=datetime.now(timezone.utc).isoformat())
            save()
        actual = digest(partial, 'md5')
        if actual != expected_md5:
            raise ValueError(f'Full archive MD5 mismatch: expected {expected_md5}, received {actual}.')
        audit.update(state='verified', actual_md5=actual, sha256=digest(partial, 'sha256'),
                     finished_utc=datetime.now(timezone.utc).isoformat())
        partial.replace(destination)
        return audit
    except Exception as exc:
        audit.update(state='failed', error=str(exc))
        raise
    finally:
        save()
        lock.unlink()


def main() -> None:
    from torchvision.datasets import CIFAR100

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    args = parser.parse_args()
    result = download_archive(CIFAR100.url.replace('http://', 'https://'),
                              args.data_root / CIFAR100.filename, CIFAR100.tgz_md5,
                              Path('logs/cifar-download-recovery.json'))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
