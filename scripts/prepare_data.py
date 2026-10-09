"""Download and prepare the one shared split, without fitting a model."""
import argparse
import json
from pathlib import Path
from common.data import prepare_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    args = parser.parse_args()
    _, _, metadata = prepare_data(args.data_root)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
