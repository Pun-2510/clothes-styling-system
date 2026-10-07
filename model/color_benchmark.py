"""Color-aware retrieval experiment. See COLOR_BENCHMARK.md; no training performed."""
import argparse
import base64
import csv
import gc
import hashlib
import html
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from compare_models import MODEL_NAMES, load_encoder, sync
from mini_fashion import dataset_paths

ROOT = Path(__file__).resolve().parent
# Broad color families, not exact shades. Multiple families stay multilabel.
ALIASES = {
    'black': ['black'], 'white': ['white', 'ivory'], 'gray': ['gray', 'grey', 'charcoal'],
    'blue': ['blue', 'navy', 'teal', 'turquoise'], 'red': ['red', 'maroon', 'burgundy'],
    'green': ['green', 'olive'], 'yellow': ['yellow', 'mustard'], 'orange': ['orange', 'peach'],
    'pink': ['pink', 'fuchsia', 'magenta'], 'purple': ['purple', 'violet', 'lavender'],
    'brown': ['brown', 'tan', 'coffee'], 'beige': ['beige', 'cream', 'khaki'],
    'gold': ['gold', 'golden'], 'silver': ['silver'],
}


def title_colors(title):
    return tuple(sorted(color for color, words in ALIASES.items()
                        if re.search(r'\b(?:' + '|'.join(words) + r')\b', str(title).lower())))


def parse_colors(value):
    colors = tuple(sorted(set(part.strip().lower() for part in str(value).split('|') if part.strip())))
    if not colors or not set(colors) <= set(ALIASES):
        raise ValueError(f'Invalid color tag: {value!r}; use canonical colors separated by |')
    return colors


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def base_text(row):
    return '. '.join(f'{label}: {row[key]}' for label, key in
                     [('Product', 'product_name'), ('Category', 'category'), ('Description', 'description')]
                     if str(row.get(key, '')).strip())


def rrf_scores(image_scores, text_scores, eligible):
    """Same equal-weight RRF(k=60) as combine_model/test.py, full eligible rankings.

    Use catalog index/product identity, never product names, to avoid merging distinct items.
    """
    candidates = np.flatnonzero(eligible)
    scores = np.zeros(len(eligible), dtype=np.float64)
    for values in (image_scores, text_scores):
        ranking = candidates[np.argsort(-values[candidates], kind='stable')]
        scores[ranking] += 1 / (60 + np.arange(1, len(ranking) + 1))
    return scores


def evaluate(score_matrix, products, queries, ks, color_filter=False):
    ids = products.product_id.to_numpy()
    categories = products.category.to_numpy()
    tags = [set(value.split('|')) for value in products.color_tags]
    totals = {k: dict(precision=0., recall=0., hit_rate=0., mrr=0., color_precision=0., returned=0.) for k in ks}
    details = []
    for query in queries:
        i, color = query['index'], query['color']
        eligible = ids != ids[i]  # Exclude ALL rows of the query product.
        same_color = np.array([color in values for values in tags])
        positive = eligible & same_color & (categories == categories[i])
        if not positive.any():
            raise ValueError('Query set must have a positive before evaluation')
        candidates = np.flatnonzero(eligible & same_color if color_filter else eligible)
        ranked = candidates[np.argsort(-score_matrix[i, query['color_index']][candidates], kind='stable')[:max(ks)]]
        for k in ks:
            selected = ranked[:k]
            hits = positive[selected]
            positions = np.flatnonzero(hits)
            # A short filtered result list must not artificially inflate precision.
            slots = min(k, int(eligible.sum()))
            values = dict(precision=float(hits.sum()) / slots,
                          recall=float(hits.sum()) / int(positive.sum()),
                          hit_rate=float(hits.any()),
                          mrr=1 / (int(positions[0]) + 1) if len(positions) else 0.,
                          color_precision=float(same_color[selected].sum()) / slots,
                          returned=float(len(selected)))
            for key, value in values.items():
                totals[k][key] += value
            details.append(dict(query_id=str(ids[i]), color=color, category=str(categories[i]), k=k,
                                positives=int(positive.sum()), **values,
                                result_ids='|'.join(ids[selected])))
    return {k: {key: value / len(queries) for key, value in totals[k].items()} for k in ks}, details


def query_set(products):
    queries, skipped = [], 0
    for i, row in products.iterrows():
        # One requested color per product: deterministic seed, to avoid overweighting multicolor items.
        colors = row.color_tags.split('|')
        color = colors[int(hashlib.sha256(row.product_id.encode()).hexdigest(), 16) % len(colors)]
        valid = ((products.product_id != row.product_id) & (products.category == row.category)
                 & products.color_tags.map(lambda value: color in value.split('|')))
        if valid.any():
            queries.append(dict(index=i, color=color, color_index=0,
                                text=f'{color} {row.category}'))
        else:
            skipped += 1
    return queries, skipped


def prepare(args, output):
    products = pd.read_csv(args.csv, dtype=str).fillna('')
    required = ['product_id', 'category', 'product_name', 'image_path']
    if not set(required) <= set(products):
        raise ValueError(f'CSV requires {required}')
    if args.split != 'all':
        if 'split' not in products:
            raise ValueError('No split column; pass --split all explicitly to use the whole catalog')
        products = products[products.split == args.split].copy()
    if products.empty or products[required].apply(lambda col: col.str.strip().eq('').any()).any():
        raise ValueError('Empty split or missing required values')
    if products.product_id.duplicated().any():
        raise ValueError('Use one row per product_id for this color experiment')
    input_rows = len(products)
    products['color_tags'] = products.product_name.map(lambda value: '|'.join(title_colors(value)))
    products['color_source'] = 'title_rule_unverified'
    products['reviewed'] = 'false'
    if args.labels:
        labels = pd.read_csv(args.labels, dtype=str).fillna('')
        if not {'product_id', 'color_tags', 'reviewed'} <= set(labels) or labels.product_id.duplicated().any():
            raise ValueError('Labels need unique product_id, color_tags, reviewed columns')
        labels = labels[labels.reviewed.str.lower().eq('true')].set_index('product_id')
        if not set(labels.index) <= set(products.product_id):
            raise ValueError('Reviewed labels contain IDs outside the requested split')
        products = products[products.product_id.isin(labels.index)].copy()
        products['color_tags'] = products.product_id.map(labels.color_tags).map(lambda value: '|'.join(parse_colors(value)))
        products['color_source'], products['reviewed'] = 'human_reviewed', 'true'
    else:
        products = products[products.color_tags.ne('')].copy()
    if len(products) < 2:
        raise ValueError('Need at least two products with usable color labels')
    def path_for(value):
        if args.image_dir:
            return args.image_dir.resolve() / value.replace('\\', '/').split('/')[-1]
        path = Path(value)
        return path if path.is_absolute() else args.csv.resolve().parent / path
    products['image_path'] = products.image_path.map(path_for)
    hashes = []
    for path in products.image_path:
        with Image.open(path) as image:
            image.verify()
        hashes.append(sha(path))
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate image bytes: deduplicate first')
    products['image_sha256'] = hashes
    # Shuffle once so stable ranking ties cannot exploit category-sorted source rows.
    products = products.sample(frac=1, random_state=42).reset_index(drop=True)
    products['product_text'] = products.apply(base_text, axis=1)
    products.to_csv(output / 'color_catalog.csv', index=False, encoding='utf-8-sig')
    products[['product_id', 'product_name', 'category', 'color_tags', 'reviewed', 'image_path']].to_csv(
        output / 'color_labels_review.csv', index=False, encoding='utf-8-sig')
    cards = []
    for row in products.itertuples():
        with Image.open(row.image_path) as image:
            image = image.convert('RGB')
            image.thumbnail((140, 140))
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
        data = base64.b64encode(buffer.getvalue()).decode()
        cards.append(f'<article><img alt="" src="data:image/jpeg;base64,{data}"><b>{html.escape(row.product_id)}</b>'
                     f'<p>{html.escape(row.product_name)}</p><strong>{html.escape(row.color_tags)}</strong></article>')
    (output / 'color_review.html').write_text('<!doctype html><meta charset="utf-8"><title>Kiểm tra nhãn màu</title>'
        '<style>body{font:14px system-ui;margin:24px;background:#faf8ff}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px}'
        'article{background:white;border:1px solid #ddd;padding:12px;border-radius:12px}img{display:block;height:140px;max-width:100%;object-fit:contain;margin:auto}'
        'b{display:block}strong{color:#703ca5}</style><h1>Kiểm tra nhãn màu</h1><p>Sửa color_tags trong color_labels_review.csv '
        'và đặt reviewed=true sau khi đối chiếu ảnh. HTML này chỉ để xem, không tự lưu nhãn.</p><main>' + ''.join(cards) + '</main>', encoding='utf-8')
    return products, input_rows


def main():
    paths = dataset_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=paths['experiment'])
    parser.add_argument('--image-dir', type=Path)
    parser.add_argument('--labels', type=Path, help='Only rows reviewed=true are evaluated')
    parser.add_argument('--split', default='test')
    parser.add_argument('--models', nargs='+', choices=[*MODEL_NAMES, 'combine', 'combine_trained'],
                        default=[*MODEL_NAMES, 'combine', 'combine_trained'])
    parser.add_argument('--resnet-checkpoint', type=Path, default=ROOT / 'Web_Test/models/resnet_outfit.pth')
    parser.add_argument('--clip-checkpoint', type=Path, default=paths['clip_checkpoint'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--ks', type=int, nargs='+', default=[1, 5, 10])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.image_dir is None and args.csv.resolve() == paths['experiment'].resolve():
        args.image_dir = paths['images']
    if min(args.batch_size, *args.ks) < 1:
        parser.error('batch-size and ks must be positive')
    args.ks = sorted(set(args.ks))
    output = args.output or ROOT / 'color_results' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    products, input_rows = prepare(args, output)
    queries, skipped = query_set(products)
    if not queries:
        raise ValueError('No query has another product matching both category and color')
    pd.DataFrame(queries).assign(product_id=[products.product_id.iloc[q['index']] for q in queries]).to_csv(output / 'queries.csv', index=False)
    report = dict(status='prepared', created_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=sha(args.csv), labels_sha256=sha(args.labels) if args.labels else None,
                  settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  input_rows=input_rows, tagged_rows=len(products), excluded_rows=input_rows-len(products),
                  evaluated_queries=len(queries), skipped_no_positive=skipped,
                  colors={c: sum(c in value.split('|') for value in products.color_tags) for c in ALIASES},
                  label_quality='human_reviewed' if args.labels else 'weak_title_labels_NOT_ground_truth',
                  models={}, errors={})
    rows, details, scores = [], [], {}

    def save():
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if rows:
            pd.DataFrame(rows).to_csv(output / 'comparison.csv', index=False)
            pd.DataFrame(details).to_csv(output / 'per_query.csv', index=False)
        lines = ['# Thử nghiệm tag màu sắc', '', f"Trạng thái: {report['status']}. Nhãn: {report['label_quality']}.",
                 f'Đầu vào {input_rows}; có nhãn {len(products)}; query hợp lệ {len(queries)}; bỏ qua {skipped} query không có đáp án khác.', '',
                 'Precision/Recall/Hit/Color Precision bên dưới là %. MRR từ 0–1.', '',
                 '| Model | Input | Variant | K | Precision | Recall | Hit | MRR | Color Precision |',
                 '|---|---|---|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            lines.append(f"| {row['model']} | {row['input']} | {row['variant']} | {row['k']} | "
                         f"{row['precision']*100:.2f} | {row['recall']*100:.2f} | {row['hit_rate']*100:.2f} | "
                         f"{row['mrr']:.4f} | {row['color_precision']*100:.2f} |")
        lines += ['', '## Cách hiểu và giới hạn', '',
                  '- Đáp án đúng: sản phẩm KHÁC có cùng danh mục và chứa màu yêu cầu. Loại chính sản phẩm query khỏi mọi ranking.',
                  '- Nhãn tự động chỉ trích từ tên sản phẩm, chưa xác nhận màu nhìn thấy trong ảnh; không phải accuracy nhận diện màu.',
                  '- Query văn bản là mẫu tiếng Anh “color category”, không dùng tên/mô tả đầy đủ của sản phẩm query; chưa đo truy vấn người dùng/tiếng Việt.',
                  '- Chỉ đánh giá catalog có nhãn; không suy rộng sang toàn bộ dữ liệu. Mỗi sản phẩm có một màu query được chọn cố định.',
                  '- baseline: văn bản gốc; text_tags: thêm tag ở đầu văn bản catalog; tag_filter: baseline + lọc nhãn màu.',
                  '- Màu có thể đã xuất hiện trong văn bản gốc. text_tags đo tác dụng của tag có cấu trúc, không phải so với dữ liệu hoàn toàn không có màu.',
                  '- tag_filter dùng cùng nhãn cho lọc và chấm điểm: chỉ đo lợi ích của metadata, KHÔNG chứng minh model học màu tốt hơn.',
                  '- ResNet dùng ảnh; SBERT dùng text; CLIP báo riêng image, text→image và text→text. Combine dùng cả ảnh + text.',
                  '- Combine là RRF(k=60) bằng trọng số, ranking đầy đủ của ResNet và SBERT trên cùng catalog; không phải phép đo nguyên trạng ứng dụng combine cũ.',
                  '- Precision dùng min(K, số ứng viên trước lọc); thiếu kết quả sau lọc tính là thiếu slot, không nâng điểm giả.',
                  '- Chưa xác nhận tập train của mọi checkpoint độc lập với test. Không huấn luyện hoặc chỉnh trọng số.',
                  '- encode_seconds là thời gian tạo embedding cho lần thử, không phải latency query/API.']
        if report['errors']:
            lines += ['', '## Lỗi', *[f'- {key}: {value}' for key, value in report['errors'].items()]]
        (output / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def record(name, task, variant, matrix):
        metric, samples = evaluate(matrix, products, queries, args.ks, color_filter=variant == 'tag_filter')
        for k, values in metric.items():
            rows.append(dict(model=name, input=task, variant=variant, k=k, **values, evaluated_queries=len(queries)))
        details.extend(dict(model=name, input=task, variant=variant, **sample) for sample in samples)

    save()
    print(f'Catalog: {len(products)}/{input_rows}; queries: {len(queries)}; output: {output.resolve()}', flush=True)
    if args.prepare_only:
        return
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    torch.manual_seed(42)
    report['environment'] = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, threads=torch.get_num_threads())
    report['status'] = 'running'
    needed = set(args.models) & set(MODEL_NAMES)
    if 'combine' in args.models:
        needed.update(['resnet18', 'sbert'])
    if 'combine_trained' in args.models:
        needed.update(['resnet18_trained', 'sbert'])
    query_products = products.copy()
    query_products['product_text'] = ''
    for query in queries:
        query_products.loc[query['index'], 'product_text'] = query['text']
    tagged = products.copy()
    tagged['product_text'] = 'Color: ' + tagged.color_tags.str.replace('|', ', ', regex=False) + '. ' + tagged.product_text
    for name in MODEL_NAMES:
        if name not in needed:
            continue
        model = encoder = None
        try:
            print(f'Encoding {name}...', flush=True)
            model, encoder, modalities, source = load_encoder(name, args)
            start = time.perf_counter()
            def encode(table, modality):
                return np.concatenate([encoder(table.iloc[i:i+args.batch_size], modality)
                                       for i in range(0, len(table), args.batch_size)])
            features = {modality: encode(products, modality) for modality in modalities}
            matrices = {}
            if 'image' in modalities:
                matrices['image'] = (features['image'] @ features['image'].T)[:, None, :]
            if 'text' in modalities:
                q = encode(query_products, 'text')
                matrices['text'] = (q @ features['text'].T)[:, None, :]
                matrices['text_tags'] = (q @ encode(tagged, 'text').T)[:, None, :]
                if 'image' in modalities:
                    matrices['text_to_image'] = (q @ features['image'].T)[:, None, :]
            sync(args.device)
            report['models'][name] = dict(source=source, encode_seconds=time.perf_counter()-start,
                                         parameters=sum(p.numel() for p in model.parameters()),
                                         checkpoint_sha256=sha(args.resnet_checkpoint) if name == 'resnet18_trained' else
                                         sha(args.clip_checkpoint / 'model.safetensors') if name == 'clip_finetuned' else None)
            scores[name] = matrices
            if name in args.models:
                for task, matrix in matrices.items():
                    if task == 'text_tags':
                        record(name, 'text', 'text_tags', matrix)
                    else:
                        record(name, task, 'baseline', matrix)
                        record(name, task, 'tag_filter', matrix)
        except Exception as error:
            report['errors'][name] = f'{type(error).__name__}: {error}'
            print(f'FAILED {name}: {error}', flush=True)
        finally:
            del model, encoder
            gc.collect()
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            save()
    for combined, resnet in [('combine', 'resnet18'), ('combine_trained', 'resnet18_trained')]:
        if combined not in args.models:
            continue
        if resnet not in scores or 'sbert' not in scores:
            report['errors'][combined] = 'Missing ResNet or SBERT scores; see encoder errors'
            continue
        for variant, text_key in [('baseline', 'text'), ('text_tags', 'text_tags')]:
            fused = np.zeros_like(scores[resnet]['image'], dtype=np.float64)
            for query in queries:
                i = query['index']
                eligible = products.product_id.to_numpy() != products.product_id.iloc[i]
                fused[i, 0] = rrf_scores(scores[resnet]['image'][i, 0], scores['sbert'][text_key][i, 0], eligible)
            record(combined, 'image+text', variant, fused)
            if variant == 'baseline':
                record(combined, 'image+text', 'tag_filter', fused)
        report['models'][combined] = dict(components=[resnet, 'sbert'], fusion='equal_weight_RRF_60')
    report['status'] = 'partial_failure' if report['errors'] else 'complete'
    save()
    print(f"Finished: {report['status']}; {output.resolve()}", flush=True)
    if report['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
