"""Archive integrity and retry boundaries; no network or dataset loading."""
import hashlib
import json

import pytest

from scripts import prefetch_cifar100 as downloader


def setup_archive(tmp_path):
    data = b'official immutable archive payload'
    target = tmp_path / 'archive.tar.gz'
    progress = tmp_path / 'audit.json'
    return data, target, progress


def test_resume_keeps_prefix_and_promotes_only_complete_verified_archive(tmp_path, monkeypatch):
    data, target, progress = setup_archive(tmp_path)
    target.write_bytes(data[:5])
    calls = []

    def fetch(url, start, end, total=None):
        calls.append((start, end))
        return data[start:end + 1], len(data)

    monkeypatch.setattr(downloader, 'fetch_range', fetch)
    result = downloader.download_archive('https://official.test/file', target,
                                          hashlib.md5(data).hexdigest(), progress, chunk_bytes=7)
    assert calls[0] == (0, 0) and calls[1][0] == 5
    assert target.read_bytes() == data
    assert result['state'] == 'verified' and result['initial_bytes'] == 5
    assert json.loads(progress.read_text())['sha256'] == hashlib.sha256(data).hexdigest()
    assert not list(tmp_path.glob('*.partial')) and not list(tmp_path.glob('*.lock'))


def test_wrong_md5_never_exposes_archive_as_ready(tmp_path, monkeypatch):
    data, target, progress = setup_archive(tmp_path)
    monkeypatch.setattr(downloader, 'fetch_range', lambda url, start, end, total=None: (data[start:end + 1], len(data)))
    with pytest.raises(ValueError, match='MD5 mismatch'):
        downloader.download_archive('https://official.test/file', target, '0' * 32, progress)
    assert not target.exists()
    assert target.with_name(target.name + '.partial').read_bytes() == data
    assert json.loads(progress.read_text())['state'] == 'failed'


def test_unexpected_http_range_is_rejected_before_reading(monkeypatch):
    class WrongRange:
        status = 206
        headers = {'Content-Range': 'bytes 0-9/100'}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            pytest.fail('A mismatched range must not be read or appended.')

    monkeypatch.setattr(downloader.urllib.request, 'urlopen', lambda *args, **kwargs: WrongRange())
    with pytest.raises(ValueError, match='different archive range'):
        downloader.fetch_range('https://official.test/file', 10, 19, 100)


def test_network_retries_are_bounded_and_preserve_existing_bytes(tmp_path, monkeypatch):
    data, target, progress = setup_archive(tmp_path)
    target.write_bytes(data[:5])
    attempts = []

    def fetch(url, start, end, total=None):
        if total is None:
            return data[:1], len(data)
        attempts.append(start)
        raise OSError('Interrupted response')

    monkeypatch.setattr(downloader, 'fetch_range', fetch)
    monkeypatch.setattr(downloader.time, 'sleep', lambda seconds: None)
    with pytest.raises(OSError, match='Interrupted'):
        downloader.download_archive('https://official.test/file', target,
                                    hashlib.md5(data).hexdigest(), progress, attempts=3)
    assert attempts == [5, 5, 5]
    assert target.with_name(target.name + '.partial').read_bytes() == data[:5]
    assert not target.exists()
