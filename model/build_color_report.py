"""Render a color benchmark, with paired deltas, without loading any ML models."""
import argparse
import csv
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.source / 'report.json').read_text(encoding='utf-8'))
    with (args.source / 'comparison.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    baseline = {(row['model'], row['input'], row['k']): row for row in rows if row['variant'] == 'baseline'}
    deltas = []
    for row in rows:
        if row['variant'] == 'baseline':
            continue
        base = baseline[(row['model'], row['input'], row['k'])]
        deltas.append({key: row[key] for key in ('model', 'input', 'variant', 'k')} | {
            f'{metric}_delta_pp': (float(row[metric])-float(base[metric]))*100
            for metric in ('precision', 'recall', 'hit_rate', 'color_precision')})
    if deltas:
        with (args.source / 'deltas.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(deltas[0]))
            writer.writeheader()
            writer.writerows(deltas)
    headers = ['Model', 'Đầu vào', 'Biến thể', 'K', 'Precision %', 'Recall %', 'Hit %', 'MRR', 'Đúng màu %', 'Số query']
    table = []
    for row in rows:
        cells = [row[key] for key in ('model', 'input', 'variant', 'k')]
        cells += [f'{float(row[key])*100:.2f}' for key in ('precision', 'recall', 'hit_rate')]
        cells += [f"{float(row['mrr']):.4f}", f"{float(row['color_precision'])*100:.2f}", row['evaluated_queries']]
        table.append('<tr>' + ''.join(f'<td>{html.escape(value)}</td>' for value in cells) + '</tr>')
    delta_table = []
    for row in deltas:
        if row['k'] != '5':
            continue
        cells = [row[key] for key in ('model', 'input', 'variant')]
        cells += [f'{row[key]:+.2f}' for key in ('precision_delta_pp', 'recall_delta_pp', 'color_precision_delta_pp')]
        delta_table.append('<tr>' + ''.join(f'<td>{html.escape(value)}</td>' for value in cells) + '</tr>')
    summary = (args.source / 'summary.md').read_text(encoding='utf-8')
    limits = summary.split('## Cách hiểu và giới hạn', 1)[-1]
    page = f'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Thử nghiệm tag màu — clothes styling</title><style>
body{{font:14px/1.6 system-ui;margin:32px;color:#30283d;background:#faf8fc}}h1,h2{{color:#63349c}}
.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:white;margin:20px 0}}
th,td{{padding:9px 12px;border:1px solid #e3dcea;text-align:right;white-space:nowrap}}th{{background:#eee6f7}}
td:first-child,td:nth-child(2),td:nth-child(3){{text-align:left}}pre{{white-space:pre-wrap;font:inherit;background:white;padding:20px}}
.note{{padding:16px;border-left:4px solid #8751bc;background:#f0e8f7}}@media print{{body{{margin:0;font-size:10px}}th,td{{padding:4px}}}}
</style><h1>Thử nghiệm bổ sung tag màu sắc</h1>
<p>Trạng thái: <b>{html.escape(report['status'])}</b> · {report['input_rows']} sản phẩm đầu vào · {report['tagged_rows']} có nhãn · {report['evaluated_queries']} query hợp lệ.</p>
<p class="note">Nguồn nhãn: <b>{html.escape(report['label_quality'])}</b>. Nhãn từ tên sản phẩm chưa phải nhãn màu ảnh đã kiểm chứng.
Đây là chất lượng truy hồi cùng loại và cùng màu, không phải độ chính xác phân loại màu.
tag_filter dùng nhãn để lọc và chấm điểm; mức tăng không chứng minh mô hình tự học màu tốt hơn.</p>
<h2>Chênh lệch so với baseline tại K=5 (điểm phần trăm)</h2>
<div class="table-wrap"><table><thead><tr><th>Model</th><th>Đầu vào</th><th>Biến thể</th><th>Δ Precision</th><th>Δ Recall</th><th>Δ Đúng màu</th></tr></thead>
<tbody>{''.join(delta_table)}</tbody></table></div>
<h2>Toàn bộ kết quả</h2><div class="table-wrap"><table><thead><tr>{''.join(f'<th>{value}</th>' for value in headers)}</tr></thead><tbody>{''.join(table)}</tbody></table></div>
<h2>Phương pháp và giới hạn</h2><pre>{html.escape(limits)}</pre></html>'''
    target = args.source / 'color_report.html'
    target.write_text(page, encoding='utf-8')
    print(target.resolve())


if __name__ == '__main__':
    main()
