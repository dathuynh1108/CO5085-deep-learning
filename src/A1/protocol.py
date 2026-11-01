"""Dependency-light protocol and suite discovery."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/a1_experiments.json'


def load_protocol(config=CONFIG):
    return json.loads(config.read_text(encoding='utf-8'))


def experiments():
    return load_protocol()['experiments']
