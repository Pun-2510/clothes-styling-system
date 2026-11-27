"""Evaluate frozen FashionCLIP on the existing three-seed gallery protocol."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .compare_full_gallery import (_load_and_validate_data, _annotated_rows,
                                   _annotated_category_rows)
from .compare_modalities import DEFAULT_RUN, _cache_signature, _cached_embeddings, _prepare_output
from .encoders import encode_clip_images, encode_clip_text
from .evaluation import (rank_queries_by_cosine, evaluate_query_gallery_category_rankings,
                         evaluate_query_gallery_identity_rankings)
from .fusion import clip_query_gallery_score_fusion


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    p.add_argument('--catalog-csv', type=Path, required=True)
    p.add_argument('--expected-gallery-size', type=int, default=5000)
    p.add_argument('--model', default='patrickjohncyh/fashion-clip')
    p.add_argument('--revision', default='main')
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--ks', type=int, nargs='+', default=[1,5,10])
    p.add_argument('--image-weight', type=float, default=0.5)
    p.add_argument('--text-image-weight', type=float, default=0.5)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def main():
    a = arguments()
    if a.expected_gallery_size < 2 or a.batch_size < 1 or min(a.ks) < 1:
        raise ValueError('Gallery, batch and K must be positive.')
    if not 0 <= a.image_weight <= 1 or not 0 <= a.text_image_weight <= 1:
        raise ValueError('Fusion weights must be in [0,1].')
    import sys
    from .compare_modalities import PROJECT
    sys.path.insert(0, str(PROJECT))
    from src.clip_data import file_sha256
    _, queries, gallery, positions, experiment, training, training_path = _load_and_validate_data(a, file_sha256)
    settings = {
        'model': a.model, 'revision_requested': a.revision,
        'run_dir': str(a.run_dir.resolve()), 'catalog_csv': str(a.catalog_csv.resolve()),
        'catalog_sha256': file_sha256(a.catalog_csv), 'dataset_sha256': experiment['dataset_sha256'],
        'training_json_sha256': file_sha256(training_path), 'gallery_size': len(gallery),
        'query_count': len(queries), 'query_split': 'test', 'ks': sorted(set(a.ks)),
        'image_weight': a.image_weight, 'text_image_weight': a.text_image_weight,
        'relevance': 'same category for modality rows; exact product ID for alignment rows',
        'category_in_encoder_text': False, 'category_policy': 'none (frozen raw baseline)',
    }
    if a.dry_run:
        print(json.dumps(settings, indent=2))
        return
    from huggingface_hub import HfApi
    resolved = HfApi().model_info(a.model, revision=a.revision).sha
    settings['revision_resolved'] = resolved
    source = a.model
    identity = {'architecture': 'FashionCLIP', 'source': source, 'revision': resolved,
                'category_in_text': False}
    signature = _cache_signature(gallery, identity)
    texts, text_meta = _cached_embeddings(
        a.cache_dir, 'fashionclip_text', signature,
        lambda: encode_clip_text(gallery.evaluation_text.tolist(), source, a.device,
                                 a.batch_size, revision=resolved), True)
    images, image_meta = _cached_embeddings(
        a.cache_dir, 'fashionclip_image', signature,
        lambda: encode_clip_images(gallery.image_path.tolist(), source, a.device,
                                   a.batch_size, revision=resolved), True)
    qids = queries.product_id.astype(str).to_numpy()
    gids = gallery.product_id.astype(str).to_numpy()
    qc = queries.category.astype(str).to_numpy()
    gc = gallery.category.astype(str).to_numpy()
    rankings = {
        'fashionclip_image_only': rank_queries_by_cosine(images[positions], images, qids, gids),
        'fashionclip_text_only': rank_queries_by_cosine(texts[positions], texts, qids, gids),
        'fashionclip_combined_raw': clip_query_gallery_score_fusion(
            images[positions], texts[positions], images, texts, qids, gids,
            image_weight=a.image_weight, text_image_weight=a.text_image_weight),
    }
    paired = {
        'fashionclip_image_to_text': rank_queries_by_cosine(images[positions], texts, qids, gids, False),
        'fashionclip_text_to_image': rank_queries_by_cosine(texts[positions], images, qids, gids, False),
    }
    ks = settings['ks']
    metrics = {k: evaluate_query_gallery_category_rankings(v, qids, qc, gids, gc, ks)
               for k,v in rankings.items()}
    metrics.update({k: evaluate_query_gallery_identity_rankings(v, qids, qc, gids, ks)
                    for k,v in paired.items()})
    specs = {
        'fashionclip_image_only': dict(comparison_group='fashionclip_baseline', role='frozen image encoder',
            deployed=False, input='image', formula='FashionCLIP S_II'),
        'fashionclip_text_only': dict(comparison_group='fashionclip_baseline', role='frozen text encoder',
            deployed=False, input='text', formula='FashionCLIP S_TT'),
        'fashionclip_combined_raw': dict(comparison_group='fashionclip_baseline', role='frozen multimodal fusion',
            deployed=False, input='image + text', formula='raw S_II + S_TI + S_TT fusion'),
        'fashionclip_image_to_text': dict(comparison_group='fashionclip_alignment', role='frozen image to text',
            deployed=False, input='image', formula='FashionCLIP S_IT'),
        'fashionclip_text_to_image': dict(comparison_group='fashionclip_alignment', role='frozen text to image',
            deployed=False, input='text', formula='FashionCLIP S_TI'),
    }
    rows=[]; category_rows=[]
    for group, task in [('fashionclip_baseline','same-category product retrieval'),
                        ('fashionclip_alignment','exact paired-product retrieval')]:
        group_specs={k:v for k,v in specs.items() if v['comparison_group']==group}
        rows += _annotated_rows(group_specs, metrics, ks, task)
        category_rows += _annotated_category_rows(group_specs, metrics, ks, task)
    out=_prepare_output(a.output_dir)
    pd.DataFrame(rows).to_csv(out/'comparison.csv', index=False)
    pd.DataFrame(category_rows).to_csv(out/'comparison_by_category.csv', index=False)
    (out/'experiment.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    (out/'comparison.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),
        'settings':settings,'models':{'image':image_meta,'text':text_meta},'metrics':metrics},
        indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print('Saved:', out)


if __name__=='__main__':
    main()
