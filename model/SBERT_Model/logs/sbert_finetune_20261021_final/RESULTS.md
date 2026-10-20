# Kết quả fine-tune SBERT retrieval

- Trạng thái: hoàn tất
- Model nền: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Dataset: `ashraq/fashion-product-images-small`
- Thiết bị: CPU
- Dữ liệu: 960 mẫu train, 240 mẫu validation
- Đánh giá retrieval: 60 query và 180 sản phẩm trong corpus
- Số epoch: 3
- Batch size: 12
- Learning rate: `2e-5`
- Loss: `BatchAllTripletLoss` với cosine distance
- Thời gian toàn bộ: 220.353 giây
- Checkpoint tốt nhất: epoch 2, chọn theo MRR@10

## Quá trình theo epoch

| Epoch | Train loss | Hit@1 | Hit@5 | MRR@10 | MAP@10 | Thời gian |
|---:|---:|---:|---:|---:|---:|---:|
| Baseline | - | 78.33% | 90.00% | 84.53% | 41.20% | - |
| 1 | 0.141420 | 93.33% | 100.00% | 95.69% | 83.59% | 62.711 s |
| 2 | 0.127555 | 98.33% | 98.33% | 98.57% | 97.48% | 63.236 s |
| 3 | 0.090394 | 98.33% | 98.33% | 98.54% | 98.00% | 59.091 s |

Model đầu ra dùng checkpoint epoch 2 vì có MRR@10 cao nhất. Kết quả cuối của
checkpoint này: Hit@1 = 98.33%, Hit@5 = 98.33%, Hit@10 = 100%, MRR@10 =
98.57%, MAP@10 = 97.48% và NDCG@10 = 97.98%.

## Kiểm tra với website

`Web_Test/app.py --self-test` đã chạy thành công bằng model mới. Cả ba chế độ
`text`, `image` và `combined` đều trả đúng sản phẩm kiểm tra ở hạng 1; trạng
thái tổng thể là `pass`.

## File liên quan

- `training_summary.json`: cấu hình, baseline, kết quả cuối và lịch sử epoch.
- `epoch_history.csv`: số liệu từng epoch để mở bằng Excel hoặc vẽ biểu đồ.
- `web_self_test.json`: kết quả kiểm tra text, image và combined trên website.
- `terminal_output.txt`: toàn bộ đầu ra terminal, dùng đuôi `.txt` để có thể đưa lên Git.
- `../../models/sbert_fashion_finetuned`: model tốt nhất đã lưu.

Lưu ý: đây là benchmark truy hồi cùng nhóm sản phẩm trên validation của sáu
lớp thời trang cân bằng. Không so sánh trực tiếp các số này với benchmark màu
327 query vì hai thí nghiệm dùng dữ liệu và tiêu chí đánh giá khác nhau.
