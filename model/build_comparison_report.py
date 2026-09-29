"""Generate the 484-product benchmark report; paths are configurable via --help."""
import argparse
import csv
import hashlib
import html
import json
import math
from collections import Counter
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
LABELS = {'resnet18': 'ResNet18 pretrained', 'resnet18_trained': 'ResNet18 đã train',
          'sbert': 'SBERT multilingual MiniLM-L12', 'clip': 'CLIP ViT-B/32 pretrained'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Directory containing report.json, comparison.csv and evaluation_catalog.csv')
    parser.add_argument('--output-dir', type=Path,
                        help='Output directory; default: model/reports/<source directory name> beside this script')
    parser.add_argument('--csv', type=Path,
                        help='Relocated dataset.csv; default: settings.csv recorded in report.json')
    parser.add_argument('--image-dir', type=Path,
                        help='Relocated image directory; images are matched by filename')
    parser.add_argument('--reference-csv', type=Path,
                        help='Optional original dataset.csv for comparison with experiment.json')
    return parser.parse_args()


def main():
    args = parse_args()
    SOURCE = args.source.expanduser().resolve()
    OUT = (args.output_dir or MODEL_DIR / 'reports' / SOURCE.name).expanduser().resolve()
    for name in ('report.json', 'comparison.csv', 'evaluation_catalog.csv'):
        if not (SOURCE / name).is_file():
            raise FileNotFoundError(f'Missing input file: {SOURCE / name}; check --source')
    report = json.loads((SOURCE / 'report.json').read_text(encoding='utf-8'))
    csv_path = args.csv or Path(report['settings']['csv'])
    if not csv_path.is_absolute() and args.csv is None:
        csv_path = SOURCE / csv_path
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f'Missing dataset: {csv_path}; pass --csv with its current location')
    def image_path(row):
        if args.image_dir:
            return args.image_dir.expanduser().resolve() / row['image_path'].replace('\\', '/').split('/')[-1]
        path = Path(row['image_path'])
        return path if path.is_absolute() else SOURCE / path
    rows = read_csv(SOURCE / 'comparison.csv')
    catalog = read_csv(SOURCE / 'evaluation_catalog.csv')
    source_rows = read_csv(csv_path)
    models = report['models']
    checks = []

    def check(label, condition):
        checks.append((label, bool(condition)))
        if not condition:
            raise ValueError('Kiểm tra thất bại: ' + label)

    check('Trạng thái complete; errors rỗng', report['status'] == 'complete' and not report['errors'])
    check('Đủ bốn mô hình yêu cầu', set(models) == set(report['settings']['models']) == set(LABELS))
    check('CSV nguồn khớp SHA-256 ghi trong lần chạy', sha(csv_path) == report['source_csv_sha256'])
    check('484 sản phẩm độc lập, chỉ thuộc test', len(catalog) == report['rows'] == 484
          and len({r['product_id'] for r in catalog}) == 484 and {r['split'] for r in catalog} == {'test'})
    original_test = {r['product_id']: r for r in source_rows if r['split'] == 'test'}
    check('Catalog khớp nội dung và thứ tự tập test',
          list(original_test) == [r['product_id'] for r in catalog]
          and all(all(r[k] == original_test[r['product_id']][k]
              for k in ('category', 'product_name', 'description', 'product_text', 'image_sha256')) for r in catalog))
    check('484 file ảnh tồn tại và khớp hash lưu trong catalog',
          all(image_path(r).is_file() and sha(image_path(r)) == r['image_sha256'] for r in catalog))
    counts = Counter(r['category'] for r in catalog)
    skipped = sum(count for count in counts.values() if count == 1)
    check('88 danh mục; 45 query không có sản phẩm cùng danh mục khác', len(counts) == 88 and skipped == 45)
    expected = {(name, task, k) for name, model in models.items() for task in model['metrics'] for k in (1, 5, 10)}
    check('Đủ 21 dòng kết quả, không trùng khóa model/task/K', len(rows) == len(expected) == 21
          and {(r['model'], r['task'], int(r['k'])) for r in rows} == expected)
    for row in rows:
        model = models[row['model']]
        metric = model['metrics'][row['task']]
        k = int(row['k'])
        for key in ('precision', 'recall', 'hit_rate', 'mrr'):
            value = float(row[key])
            assert 0 <= value <= 1 and math.isclose(value, metric[f'{key}@{k}'], abs_tol=1e-12)
        assert int(row['parameters']) == model['parameters']
        assert math.isclose(float(row['parameter_mib']), model['parameter_mib'])
        assert int(row['evaluated_queries']) == metric['evaluated_queries']
        assert int(row['skipped_queries']) == metric['skipped_queries']
        assert metric['evaluated_queries'] + metric['skipped_queries'] == 484
        if row['task'].endswith('product'):
            assert math.isclose(float(row['precision']) * k, float(row['recall']), abs_tol=1e-10)
            assert not row['query_p50_ms'] and not row['query_p95_ms']
        else:
            modality = row['task'].split('_')[0]
            assert metric['evaluated_queries'] == 439 and metric['skipped_queries'] == 45
            for key in ('query_p50_ms', 'query_p95_ms'):
                assert math.isclose(float(row[key]), model['modalities'][modality][key])
    check('Toàn bộ metric, tham số, số query và độ trễ trong CSV khớp JSON', True)
    check('Quan hệ Precision@K × K = Recall@K đúng cho truy hồi một sản phẩm', True)
    for model in models.values():
        for modality in model['modalities'].values():
            assert math.isclose(modality['embedding_mib'], 484 * modality['embedding_dim'] * 4 / 2**20)
            assert math.isclose(modality['catalog_items_per_second'], 484 / modality['catalog_encode_seconds'])
    check('Dung lượng embedding và thông lượng nhất quán', True)

    # The benchmark CSV hash differs from the old experiment record; determine whether content changed.
    experiment_path = csv_path.parent / 'experiment.json'
    experiment = json.loads(experiment_path.read_text(encoding='utf-8')) if experiment_path.is_file() else None
    reference = args.reference_csv.expanduser().resolve() if args.reference_csv else None
    same_content = read_csv(reference) == source_rows if reference else None
    reference_matches_old = sha(reference) == experiment['dataset_sha256'] if reference and experiment else None
    audit = {'source': str(SOURCE), 'output': str(OUT), 'dataset_csv': str(csv_path),
        'reference_csv': str(reference) if reference else None,
        'image_dir': str(args.image_dir.resolve()) if args.image_dir else None,
        'checks': checks, 'categories': dict(counts),
        'source_files_sha256': {name: sha(SOURCE / name) for name in ('report.json', 'comparison.csv', 'evaluation_catalog.csv')},
        'old_experiment_hash_matches_current_csv': experiment['dataset_sha256'] == sha(csv_path) if experiment else None,
        'reference_matches_old_experiment_hash': reference_matches_old,
        'parsed_dataset_matches_reference': same_content}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    md, web = [], []

    def heading(text, level=2):
        md.extend(['#' * level + ' ' + text, ''])
        web.append(f'<h{level}>{html.escape(text)}</h{level}>')

    def paragraph(text):
        md.extend([text, ''])
        web.append('<p>' + html.escape(text) + '</p>')

    def table(headers, data):
        md.append('| ' + ' | '.join(headers) + ' |')
        md.append('| ' + ' | '.join('---' for _ in headers) + ' |')
        md.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in data)
        md.append('')
        web.append('<div class="table"><table><thead><tr>' + ''.join('<th>' + html.escape(h) + '</th>' for h in headers) + '</tr></thead><tbody>')
        for row in data:
            web.append('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in row) + '</tr>')
        web.append('</tbody></table></div>')

    heading('Báo cáo so sánh ResNet18, SBERT và CLIP', 1)
    paragraph('Lần chạy: 29/09/2026, 22:17:46 (UTC+7). Báo cáo tổng hợp từ kết quả đã lưu; không chạy lại suy luận mô hình. '
              'Đề xuất CLIP làm mô hình nền cho project cần cả ảnh và văn bản, dựa trên khả năng truy hồi chéo và kết quả theo tác vụ. '
              'Không kết luận CLIP tốt nhất ở mọi tiêu chí; chưa đánh giá CLIP fine-tuned trong lần chạy này.')
    heading('1. Kiểm tra kết quả và thiết lập thực nghiệm')
    table(['Hạng mục', 'Kết quả'], [(label, 'Đạt') for label, ok in checks])
    paragraph('Kiểm tra trên xác nhận tính nhất quán của các file và ảnh đầu vào, không thay thế việc tái chạy để xác minh thứ hạng từ embedding. '
              'Embedding và độ trễ từng query không được lưu nên chưa thể tính lại metric hoặc khoảng tin cậy từ các file này.')
    if audit['old_experiment_hash_matches_current_csv'] is False:
        paragraph('Lưu ý nguồn dữ liệu: hash dataset.csv hiện tại khớp report.json nhưng khác hash trong experiment.json cũ. '
                  f'Đối chiếu bản tham chiếu: nội dung CSV sau parse giống nhau = {same_content}; hash bản tham chiếu khớp experiment.json cũ = {reference_matches_old}. '
                  'Giá trị None nghĩa là chưa cung cấp đủ file để đối chiếu. Không suy ra nguyên nhân khác hash chỉ từ kết quả này. '
                  'Không tự sửa metadata cũ; chỉ dùng danh sách và hash được lưu cho lần benchmark hiện tại.')
    elif experiment is None:
        paragraph('Không có experiment.json cạnh CSV nguồn; chưa đối chiếu metadata thí nghiệm cũ. CSV vẫn đã được kiểm tra với hash của lần benchmark.')
    table(['Thông số', 'Giá trị'], [
        ['Thiết bị', 'CPU; Intel64 Family 6 Model 186 Stepping 2; 14 PyTorch threads'],
        ['Môi trường', 'Windows; Python 3.11.9; PyTorch 2.13.0; NumPy 2.4.6; pandas 3.0.6'],
        ['Catalog / tập đánh giá', '484 sản phẩm test, 88 danh mục; gallery chỉ gồm 484 sản phẩm này'],
        ['Đánh giá cùng danh mục', '439 query hợp lệ; bỏ 45 query không có positive sau khi loại sản phẩm truy vấn'],
        ['Đánh giá chéo phương thức', '484 query; cùng product_id là positive'],
        ['Batch size / K', '16 / 1, 5, 10'],
        ['Đo độ trễ', '30 query sau warm-up, seed chọn query 42, batch truy vấn = 1'],
        ['ResNet18 pretrained', models['resnet18']['source']],
        ['ResNet18 đã train', models['resnet18_trained']['source']],
        ['SBERT', models['sbert']['source']], ['CLIP', models['clip']['source']]])
    heading('2. Ý nghĩa chỉ số')
    table(['Chỉ số', 'Định nghĩa / đơn vị'], [
        ['Precision@K', 'Số positive trong Top-K / số kết quả trả về; bảng biểu diễn %'],
        ['Recall@K', 'Số positive trong Top-K / tổng positive trong gallery; bảng biểu diễn %'],
        ['Hit Rate@K', 'Tỷ lệ query có ít nhất một positive trong Top-K; %'],
        ['MRR@K', 'Trung bình nghịch đảo vị trí positive đầu tiên trong Top-K; không có thì 0; thang 0–1'],
        ['P50 / P95', 'Phân vị 50% / 95% độ trễ truy vấn; ms; càng nhỏ càng tốt'],
        ['Macro average', 'Mỗi query hợp lệ có trọng số bằng nhau; không phải trung bình đều theo danh mục']])
    paragraph('Ở bài toán cùng danh mục có nhiều positive, Recall@1 thấp không đồng nghĩa Top-1 thường sai. '
              'Ví dụ ResNet18 pretrained có Precision@1 = 74,49% nhưng Recall@1 = 6,37%. '
              'Ở truy hồi chéo, mỗi query có đúng một positive: Recall = Hit Rate; Precision@5 = Recall@5 / 5.')
    for title, task in [
        ('3. Ảnh → ảnh: truy hồi cùng danh mục', 'image_to_image_category'),
        ('4. Văn bản → văn bản: truy hồi cùng danh mục', 'text_to_text_category'),
        ('5. Văn bản → ảnh: truy hồi đúng sản phẩm', 'text_to_image_product'),
        ('6. Ảnh → văn bản: truy hồi đúng sản phẩm', 'image_to_text_product')]:
        heading(title)
        data = []
        for row in rows:
            if row['task'] == task:
                data.append([LABELS[row['model']], row['k'],
                    *[f'{100 * float(row[key]):.2f}' for key in ('precision', 'recall', 'hit_rate')],
                    f"{float(row['mrr']):.4f}", row['evaluated_queries']])
        table(['Mô hình', 'K', 'Precision (%)', 'Recall (%)', 'Hit Rate (%)', 'MRR', 'Query'], data)
    heading('7. Tốc độ mã hóa và tìm kiếm trên CPU')
    data = []
    for name, model in models.items():
        for modality, perf in model['modalities'].items():
            data.append([LABELS[name], 'Ảnh' if modality == 'image' else 'Văn bản',
                f"{perf['catalog_encode_seconds']:.3f}", f"{perf['catalog_items_per_second']:.2f}",
                f"{perf['query_p50_ms']:.3f}", f"{perf['query_p95_ms']:.3f}", perf['latency_samples']])
    table(['Mô hình', 'Đầu vào', 'Mã hóa 484 mẫu (s)', 'Mẫu/giây', 'P50 (ms)', 'P95 (ms)', 'Số mẫu đo'], data)
    paragraph('Độ trễ gồm đọc ảnh (nếu có), tiền xử lý, mã hóa, truyền embedding về CPU và xếp hạng cùng phương thức; '
              'không gồm dịch Việt–Anh, HTTP hoặc giao diện. Không có phép đo riêng độ trễ truy hồi chéo ảnh–văn bản. '
              'Tốc độ tạo catalog theo batch và độ trễ từng query là hai phép đo khác nhau.')
    heading('8. Tham số, dung lượng và thời gian nạp')
    table(['Mô hình', 'Số tham số', 'Tham số (MiB)', 'Chiều embedding', 'Embedding catalog (MiB)', 'Nạp model (s)'], [
        [LABELS[name], f"{model['parameters']:,}", f"{model['parameter_mib']:.2f}",
         ' / '.join(f"{mod}: {p['embedding_dim']}" for mod, p in model['modalities'].items()),
         f"{sum(p['embedding_mib'] for p in model['modalities'].values()):.3f}", f"{model['load_seconds']:.3f}"]
        for name, model in models.items()])
    paragraph('CLIP: số tham số tính cả hai encoder; embedding catalog gồm cả ảnh và văn bản (tổng 1,891 MiB). '
              'ResNet chỉ tính backbone dùng trích đặc trưng, đã bỏ lớp phân loại. '
              'Dung lượng tham số là tổng byte tensor tham số, không phải RAM đỉnh hoặc kích thước toàn bộ checkpoint. '
              'Thời gian nạp có thể gồm tải mạng/cache, không dùng để kết luận mô hình nào suy luận nhanh hơn. CPU RAM và GPU memory chưa được đo trong lần chạy CPU này.')
    heading('9. Chênh lệch đáng chú ý')
    def metric(name, task, key):
        return models[name]['metrics'][task][key]
    image_task, text_task = 'image_to_image_category', 'text_to_text_category'
    table(['So sánh', 'Chênh lệch trong lần chạy'], [
        ['CLIP − ResNet pretrained, ảnh Recall@5', f"{100*(metric('clip', image_task, 'recall@5')-metric('resnet18', image_task, 'recall@5')):+.2f} điểm phần trăm"],
        ['CLIP − ResNet pretrained, ảnh Recall@10', f"{100*(metric('clip', image_task, 'recall@10')-metric('resnet18', image_task, 'recall@10')):+.2f} điểm phần trăm"],
        ['CLIP − ResNet pretrained, ảnh Precision@1', f"{100*(metric('clip', image_task, 'precision@1')-metric('resnet18', image_task, 'precision@1')):+.2f} điểm phần trăm"],
        ['CLIP / ResNet pretrained, P50 ảnh', f"{models['clip']['modalities']['image']['query_p50_ms']/models['resnet18']['modalities']['image']['query_p50_ms']:.2f} lần (CLIP chậm hơn)"],
        ['SBERT − CLIP, văn bản Precision@1', f"{100*(metric('sbert', text_task, 'precision@1')-metric('clip', text_task, 'precision@1')):+.2f} điểm phần trăm"],
        ['SBERT − CLIP, văn bản Recall@5', f"{100*(metric('sbert', text_task, 'recall@5')-metric('clip', text_task, 'recall@5')):+.2f} điểm phần trăm"],
        ['SBERT / CLIP, P50 văn bản', f"{models['sbert']['modalities']['text']['query_p50_ms']/models['clip']['modalities']['text']['query_p50_ms']:.2f} lần (SBERT chậm hơn)"]])
    paragraph('ResNet đã train thấp hơn ResNet pretrained ở toàn bộ chỉ số chất lượng đã đo cho ảnh. '
              'Đây là quan sát trên catalog hiện tại, chưa xác định nguyên nhân. Huấn luyện phân loại sáu lớp không bảo đảm tăng chất lượng truy hồi trên 88 danh mục; '
              'cần kiểm tra dữ liệu huấn luyện và đánh giá độc lập trước khi quy kết nguyên nhân.')
    heading('10. Đề xuất lựa chọn cho project')
    table(['Nhu cầu', 'Lựa chọn được số liệu hỗ trợ', 'Đánh đổi'], [
        ['Tìm ảnh với độ trễ CPU thấp', 'ResNet18 pretrained', 'P50 21,041 ms; Precision@1 74,49%; Recall@10 thấp hơn CLIP'],
        ['Tìm mô tả cùng danh mục', 'SBERT', 'Precision@1 82,00%; Recall@10 43,52%; P50 102,311 ms'],
        ['Một mô hình hỗ trợ ảnh, văn bản và truy hồi chéo', 'CLIP pretrained làm mô hình nền', 'Text→image Recall@10 91,32%; ảnh chậm hơn ResNet trên CPU'],
        ['Chọn checkpoint triển khai CLIP fine-tuned', 'Chưa kết luận từ lần chạy này', 'Cần thêm clip_finetuned vào cùng benchmark'],
        ['So với hệ SBERT + ResNet + RRF', 'Chưa đánh giá', 'Cần benchmark riêng hệ kết hợp, không suy ra từ hai model độc lập']])
    paragraph('Đoạn kết luận đề xuất cho báo cáo: “Trên tập đánh giá gồm 484 sản phẩm, CLIP pretrained đạt Recall@10 '
              '41,93% cho truy hồi ảnh cùng danh mục và 91,32% cho truy hồi văn bản sang ảnh đúng sản phẩm. '
              'ResNet18 pretrained có ưu thế độ trễ tìm ảnh, còn SBERT có ưu thế chất lượng tìm văn bản cùng danh mục. '
              'Với yêu cầu project hỗ trợ đồng thời ảnh và mô tả, nhóm lựa chọn CLIP làm mô hình nền vì cung cấp không gian biểu diễn chung '
              'và khả năng truy hồi chéo phương thức. Việc chọn checkpoint fine-tuned cần được xác nhận bằng phép đánh giá tương ứng.”')
    heading('11. Giới hạn và điều kiện sử dụng số liệu')
    for text in [
        'Không xếp hạng SBERT và ResNet bằng cách so trực tiếp điểm của hai tác vụ khác nhau. Không tạo một điểm accuracy tổng hợp từ các bảng này.',
        'Văn bản truy vấn là prompt catalog gồm tên, danh mục và mô tả; danh mục xuất hiện ngay trong đầu vào. Kết quả cùng danh mục là phép đo có hỗ trợ metadata, chưa phản ánh truy vấn tự do của người dùng.',
        'Hai text encoder có giới hạn token khác nhau; benchmark dùng giới hạn mặc định của model. Thời gian và chất lượng chịu ảnh hưởng bởi lượng văn bản thực sự được xử lý.',
        'Danh mục là nhãn thay thế cho mức liên quan, không phải đánh giá phong cách, phối đồ hay hài lòng người dùng.',
        'Chưa xác minh độc lập train/test đối với checkpoint ResNet. Không gọi đây là bằng chứng tổng quát hóa hoàn toàn không rò rỉ dữ liệu.',
        'Một lần chạy CPU và 30 query đo độ trễ chưa đủ suy luận ý nghĩa thống kê hoặc tải đồng thời. Chưa có khoảng tin cậy, độ lệch chuẩn hoặc số đo GPU.',
        'Không đo dịch tiếng Việt, category bonus/hard filtering của ứng dụng, API hay hệ RRF. Không gộp số liệu fine-tuned cũ vào benchmark này.',
        'Không dùng thời gian nạp model hoặc dung lượng tham số làm đại diện cho độ trễ suy luận hoặc RAM thực tế.']:
        paragraph(text)
    heading('12. Nguồn và tệp kiểm chứng')
    paragraph('Thư mục đầu vào: ' + str(SOURCE))
    table(['File nguồn', 'SHA-256'], list(audit['source_files_sha256'].items()))
    paragraph('audit.json đi kèm lưu các kiểm tra; comparison_percent.csv chứa đầy đủ 21 dòng, Precision/Recall/Hit Rate đã đổi sang %, MRR giữ thang 0–1. '
              f'Báo cáo được ghi tại {OUT}; dữ liệu nguồn tại {SOURCE} không bị thay đổi.')
    (OUT / 'bao_cao_so_sanh.md').write_text('\n'.join(md), encoding='utf-8')
    css = 'body{font:15px/1.65 system-ui,sans-serif;color:#182532;max-width:1250px;margin:40px auto;padding:0 24px}h1,h2{color:#153e5a}h2{margin-top:36px}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}th,td{border:1px solid #cbd5df;padding:9px 12px;text-align:left}th{background:#eaf1f7}tr:nth-child(even){background:#f7f9fc}td{overflow-wrap:anywhere}@media print{body{font-size:10pt;margin:0;max-width:none}table{font-size:8pt}tr{break-inside:avoid}h2{break-after:avoid}.table{overflow:visible}}'
    (OUT / 'bao_cao_so_sanh.html').write_text('<!doctype html><html lang="vi"><meta charset="utf-8"><title>Báo cáo so sánh mô hình</title><style>' + css + '</style><body>' + '\n'.join(web) + '</body></html>', encoding='utf-8')
    converted = []
    for row in rows:
        converted.append({'model': LABELS[row['model']], 'task': row['task'], 'k': row['k'],
            **{key + '_percent': round(100 * float(row[key]), 4) for key in ('precision', 'recall', 'hit_rate')},
            **{key: row[key] for key in ('mrr', 'evaluated_queries', 'skipped_queries', 'parameters', 'parameter_mib', 'embedding_dim', 'query_p50_ms', 'query_p95_ms')}})
    with (OUT / 'comparison_percent.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(converted[0]))
        writer.writeheader()
        writer.writerows(converted)
    print(json.dumps({'output': str(OUT), 'checks_passed': len(checks), 'rows': len(rows),
                      'same_dataset_values': same_content, 'old_reference_hash_verified': reference_matches_old}, ensure_ascii=False))


if __name__ == '__main__':
    main()
