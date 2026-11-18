"""Download public CTU-13 Parquet files from KaggleHub into data/raw.
"""
from pathlib import Path
import shutil


def main():
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit('Install requirements first: pip install -r requirements.txt') from exc
    cache_dir = Path(kagglehub.dataset_download('dhoogla/ctu13'))
    destination = Path('data/raw')
    destination.mkdir(parents=True, exist_ok=True)
    files = sorted(cache_dir.rglob('*.parquet'))
    if not files:
        raise RuntimeError(f'No Parquet files found in downloaded dataset: {cache_dir}')
    for source in files:
        target = destination / source.name
        if target.exists():
            print(f'Already exists: {target}')
            continue
        shutil.copy2(source, target)
        print(f'Copied: {target}')
    print(f'{len(files)} Parquet files available; run: python -m src.experiment')


if __name__ == '__main__':
    main()
