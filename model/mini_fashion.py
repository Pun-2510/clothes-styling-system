"""Shared local Mini Fashion catalog, matching project/src (no Hub dataset fallback)."""
import argparse
import json
import os
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
LOCAL_CONFIG = MODEL_DIR / 'mini_fashion.local.json'


def project_dir():
    configured = os.environ.get('MINI_FASHION_PROJECT')
    if not configured and LOCAL_CONFIG.is_file():
        configured = json.loads(LOCAL_CONFIG.read_text(encoding='utf-8'))['project_dir']
    return Path(configured).expanduser().resolve() if configured else MODEL_DIR.parent / 'project'


def dataset_paths(root=None):
    root = Path(root).resolve() if root else project_dir()
    return dict(catalog=root / 'data/processed/products.csv', images=root / 'data/data',
                experiment=root / 'runs/clip_finetune_3epochs_local/dataset.csv',
                clip_checkpoint=root / 'runs/clip_finetune_3epochs_local/best')


def load_catalog(split='all', root=None):
    import pandas as pd
    paths = dataset_paths(root)
    source = paths['catalog'] if split == 'all' else paths['experiment']
    if not source.is_file():
        raise FileNotFoundError(f'Missing Mini Fashion CSV: {source}. Prepare project data first, '
                                'or configure MINI_FASHION_PROJECT / model/mini_fashion.py --project-dir ... --save-local')
    frame = pd.read_csv(source, dtype=str).fillna('')
    required = ['product_id', 'product_name', 'category', 'image_path']
    if not set(required) <= set(frame):
        raise ValueError(f'{source} needs columns {required}; use the processed project CSV, not raw data.csv')
    if split != 'all':
        if 'split' not in frame:
            raise ValueError('Experiment CSV has no split; refusing to create a different evaluation split')
        frame = frame[frame.split == split].copy()
    if frame.empty or frame[required].apply(lambda col: col.str.strip().eq('').any()).any():
        raise ValueError('Empty Mini Fashion split or blank required fields')
    if frame.product_id.duplicated().any():
        raise ValueError('Duplicate product_id in Mini Fashion CSV')

    def image_path(row):
        # Prefer the selected project's images over stale absolute paths from another machine.
        name = str(row.get('image_reference') or row.image_path).replace('\\', '/').split('/')[-1]
        local = paths['images'] / name
        original = Path(row.image_path)
        if local.is_file():
            return str(local.resolve())
        if original.is_absolute() and original.is_file():
            return str(original.resolve())
        raise FileNotFoundError(f'Mini Fashion image missing: {local}')

    frame['image_path'] = frame.apply(image_path, axis=1)
    # Compatibility aliases: keep the canonical columns/IDs for traceability.
    frame['productDisplayName'] = frame.product_name
    frame['articleType'] = frame.category
    frame['image'] = frame.image_path
    for alias, source_column in [('baseColour', 'color_tags'), ('brandName', 'brand'), ('gender', 'gender')]:
        if alias not in frame:
            frame[alias] = frame[source_column] if source_column in frame else ''
    frame.attrs['source_csv'] = str(source)
    return frame.reset_index(drop=True)


def load_training_dataset():
    """Return HF Dataset API locally for legacy training scripts; use only saved train rows."""
    from datasets import Dataset
    return Dataset.from_pandas(load_catalog('train'), preserve_index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path)
    parser.add_argument('--save-local', action='store_true', help='Remember this machine path in an ignored JSON')
    parser.add_argument('--split', default='all', choices=['all', 'train', 'validation', 'test'])
    args = parser.parse_args()
    if args.save_local and not args.project_dir:
        parser.error('--save-local requires --project-dir')
    frame = load_catalog(args.split, args.project_dir)
    if args.save_local:
        LOCAL_CONFIG.write_text(json.dumps({'project_dir': str(args.project_dir.resolve())}, indent=2), encoding='utf-8')
    print(json.dumps({'dataset': 'Mini Fashion (project)', 'source': frame.attrs['source_csv'],
                      'split': args.split, 'rows': len(frame), 'categories': frame.category.nunique()}, indent=2))


if __name__ == '__main__':
    main()
