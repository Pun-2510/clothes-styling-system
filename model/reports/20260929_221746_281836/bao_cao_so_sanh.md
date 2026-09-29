# Báo cáo so sánh ResNet18, SBERT và CLIP

Lần chạy: 29/09/2026, 22:17:46 (UTC+7). Báo cáo tổng hợp từ kết quả đã lưu; không chạy lại suy luận mô hình. Đề xuất CLIP làm mô hình nền cho project cần cả ảnh và văn bản, dựa trên khả năng truy hồi chéo và kết quả theo tác vụ. Không kết luận CLIP tốt nhất ở mọi tiêu chí; chưa đánh giá CLIP fine-tuned trong lần chạy này.

## 1. Kiểm tra kết quả và thiết lập thực nghiệm

| Hạng mục | Kết quả |
| --- | --- |
| Trạng thái complete; errors rỗng | Đạt |
| Đủ bốn mô hình yêu cầu | Đạt |
| CSV nguồn khớp SHA-256 ghi trong lần chạy | Đạt |
| 484 sản phẩm độc lập, chỉ thuộc test | Đạt |
| Catalog khớp nội dung và thứ tự tập test | Đạt |
| 484 file ảnh tồn tại và khớp hash lưu trong catalog | Đạt |
| 88 danh mục; 45 query không có sản phẩm cùng danh mục khác | Đạt |
| Đủ 21 dòng kết quả, không trùng khóa model/task/K | Đạt |
| Toàn bộ metric, tham số, số query và độ trễ trong CSV khớp JSON | Đạt |
| Quan hệ Precision@K × K = Recall@K đúng cho truy hồi một sản phẩm | Đạt |
| Dung lượng embedding và thông lượng nhất quán | Đạt |

Kiểm tra trên xác nhận tính nhất quán của các file và ảnh đầu vào, không thay thế việc tái chạy để xác minh thứ hạng từ embedding. Embedding và độ trễ từng query không được lưu nên chưa thể tính lại metric hoặc khoảng tin cậy từ các file này.

Lưu ý nguồn dữ liệu: hash dataset.csv hiện tại khớp report.json nhưng khác hash trong experiment.json cũ. Đối chiếu bản tham chiếu: nội dung CSV sau parse giống nhau = True; hash bản tham chiếu khớp experiment.json cũ = False. Hai bản CSV hiện có giống nhau về giá trị sau parse, nhưng chưa có bản khớp hash thí nghiệm cũ để xác định nguyên nhân sai khác hoặc xác nhận dữ liệu nguyên bản của thí nghiệm cũ. Không tự sửa metadata cũ; chỉ dùng danh sách và hash được lưu cho lần benchmark hiện tại.

| Thông số | Giá trị |
| --- | --- |
| Thiết bị | CPU; Intel64 Family 6 Model 186 Stepping 2; 14 PyTorch threads |
| Môi trường | Windows; Python 3.11.9; PyTorch 2.13.0; NumPy 2.4.6; pandas 3.0.6 |
| Catalog / tập đánh giá | 484 sản phẩm test, 88 danh mục; gallery chỉ gồm 484 sản phẩm này |
| Đánh giá cùng danh mục | 439 query hợp lệ; bỏ 45 query không có positive sau khi loại sản phẩm truy vấn |
| Đánh giá chéo phương thức | 484 query; cùng product_id là positive |
| Batch size / K | 16 / 1, 5, 10 |
| Đo độ trễ | 30 query sau warm-up, seed chọn query 42, batch truy vấn = 1 |
| ResNet18 pretrained | ResNet18_Weights.IMAGENET1K_V1 |
| ResNet18 đã train | C:\DACNTT\model\Web_Test\models\resnet_outfit.pth |
| SBERT | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| CLIP | openai/clip-vit-base-patch32 |

## 2. Ý nghĩa chỉ số

| Chỉ số | Định nghĩa / đơn vị |
| --- | --- |
| Precision@K | Số positive trong Top-K / số kết quả trả về; bảng biểu diễn % |
| Recall@K | Số positive trong Top-K / tổng positive trong gallery; bảng biểu diễn % |
| Hit Rate@K | Tỷ lệ query có ít nhất một positive trong Top-K; % |
| MRR@K | Trung bình nghịch đảo vị trí positive đầu tiên trong Top-K; không có thì 0; thang 0–1 |
| P50 / P95 | Phân vị 50% / 95% độ trễ truy vấn; ms; càng nhỏ càng tốt |
| Macro average | Mỗi query hợp lệ có trọng số bằng nhau; không phải trung bình đều theo danh mục |

Ở bài toán cùng danh mục có nhiều positive, Recall@1 thấp không đồng nghĩa Top-1 thường sai. Ví dụ ResNet18 pretrained có Precision@1 = 74,49% nhưng Recall@1 = 6,37%. Ở truy hồi chéo, mỗi query có đúng một positive: Recall = Hit Rate; Precision@5 = Recall@5 / 5.

## 3. Ảnh → ảnh: truy hồi cùng danh mục

| Mô hình | K | Precision (%) | Recall (%) | Hit Rate (%) | MRR | Query |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet18 pretrained | 1 | 74.49 | 6.37 | 74.49 | 0.7449 | 439 |
| ResNet18 pretrained | 5 | 64.28 | 24.13 | 89.52 | 0.8066 | 439 |
| ResNet18 pretrained | 10 | 56.77 | 37.74 | 93.62 | 0.8122 | 439 |
| ResNet18 đã train | 1 | 66.51 | 6.05 | 66.51 | 0.6651 | 439 |
| ResNet18 đã train | 5 | 60.91 | 22.60 | 86.56 | 0.7483 | 439 |
| ResNet18 đã train | 10 | 54.31 | 34.26 | 92.48 | 0.7565 | 439 |
| CLIP ViT-B/32 pretrained | 1 | 71.75 | 6.57 | 71.75 | 0.7175 | 439 |
| CLIP ViT-B/32 pretrained | 5 | 66.74 | 27.12 | 92.48 | 0.8036 | 439 |
| CLIP ViT-B/32 pretrained | 10 | 58.25 | 41.93 | 95.22 | 0.8076 | 439 |

## 4. Văn bản → văn bản: truy hồi cùng danh mục

| Mô hình | K | Precision (%) | Recall (%) | Hit Rate (%) | MRR | Query |
| --- | --- | --- | --- | --- | --- | --- |
| SBERT multilingual MiniLM-L12 | 1 | 82.00 | 8.02 | 82.00 | 0.8200 | 439 |
| SBERT multilingual MiniLM-L12 | 5 | 70.98 | 28.81 | 94.08 | 0.8696 | 439 |    
| SBERT multilingual MiniLM-L12 | 10 | 62.92 | 43.52 | 96.58 | 0.8731 | 439 |
| CLIP ViT-B/32 pretrained | 1 | 76.77 | 7.33 | 76.77 | 0.7677 | 439 |
| CLIP ViT-B/32 pretrained | 5 | 69.29 | 26.84 | 93.85 | 0.8393 | 439 |
| CLIP ViT-B/32 pretrained | 10 | 60.66 | 42.05 | 96.13 | 0.8423 | 439 |

## 5. Văn bản → ảnh: truy hồi đúng sản phẩm

| Mô hình | K | Precision (%) | Recall (%) | Hit Rate (%) | MRR | Query |
| --- | --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/32 pretrained | 1 | 47.52 | 47.52 | 47.52 | 0.4752 | 484 |
| CLIP ViT-B/32 pretrained | 5 | 16.28 | 81.40 | 81.40 | 0.6031 | 484 |
| CLIP ViT-B/32 pretrained | 10 | 9.13 | 91.32 | 91.32 | 0.6166 | 484 |

## 6. Ảnh → văn bản: truy hồi đúng sản phẩm

| Mô hình | K | Precision (%) | Recall (%) | Hit Rate (%) | MRR | Query |
| --- | --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/32 pretrained | 1 | 49.59 | 49.59 | 49.59 | 0.4959 | 484 |
| CLIP ViT-B/32 pretrained | 5 | 16.28 | 81.40 | 81.40 | 0.6140 | 484 |
| CLIP ViT-B/32 pretrained | 10 | 9.21 | 92.15 | 92.15 | 0.6290 | 484 |

## 7. Tốc độ mã hóa và tìm kiếm trên CPU

| Mô hình | Đầu vào | Mã hóa 484 mẫu (s) | Mẫu/giây | P50 (ms) | P95 (ms) | Số mẫu đo |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet18 pretrained | Ảnh | 7.558 | 64.04 | 21.041 | 24.731 | 30 |
| ResNet18 đã train | Ảnh | 7.176 | 67.45 | 23.367 | 29.988 | 30 |
| SBERT multilingual MiniLM-L12 | Văn bản | 25.383 | 19.07 | 102.311 | 111.314 | 30 |
| CLIP ViT-B/32 pretrained | Ảnh | 39.601 | 12.22 | 134.764 | 153.010 | 30 |
| CLIP ViT-B/32 pretrained | Văn bản | 24.361 | 19.87 | 37.417 | 48.137 | 30 |

Độ trễ gồm đọc ảnh (nếu có), tiền xử lý, mã hóa, truyền embedding về CPU và xếp hạng cùng phương thức; không gồm dịch Việt–Anh, HTTP hoặc giao diện. Không có phép đo riêng độ trễ truy hồi chéo ảnh–văn bản. Tốc độ tạo catalog theo batch và độ trễ từng query là hai phép đo khác nhau.

## 8. Tham số, dung lượng và thời gian nạp

| Mô hình | Số tham số | Tham số (MiB) | Chiều embedding | Embedding catalog (MiB) | Nạp model (s) |
| --- | --- | --- | --- | --- | --- |
| ResNet18 pretrained | 11,176,512 | 42.64 | image: 512 | 0.945 | 2.539 |
| ResNet18 đã train | 11,176,512 | 42.64 | image: 512 | 0.945 | 0.174 |
| SBERT multilingual MiniLM-L12 | 117,653,760 | 448.81 | text: 384 | 0.709 | 36.045 |
| CLIP ViT-B/32 pretrained | 151,277,313 | 577.08 | image: 512 / text: 512 | 1.891 | 78.087 |

CLIP: số tham số tính cả hai encoder; embedding catalog gồm cả ảnh và văn bản (tổng 1,891 MiB). ResNet chỉ tính backbone dùng trích đặc trưng, đã bỏ lớp phân loại. Dung lượng tham số là tổng byte tensor tham số, không phải RAM đỉnh hoặc kích thước toàn bộ checkpoint. Thời gian nạp có thể gồm tải mạng/cache, không dùng để kết luận mô hình nào suy luận nhanh hơn. CPU RAM và GPU memory chưa được đo trong lần chạy CPU này.

## 9. Chênh lệch đáng chú ý

| So sánh | Chênh lệch trong lần chạy |
| --- | --- |
| CLIP − ResNet pretrained, ảnh Recall@5 | +2.99 điểm phần trăm |
| CLIP − ResNet pretrained, ảnh Recall@10 | +4.19 điểm phần trăm |
| CLIP − ResNet pretrained, ảnh Precision@1 | -2.73 điểm phần trăm |
| CLIP / ResNet pretrained, P50 ảnh | 6.40 lần (CLIP chậm hơn) |
| SBERT − CLIP, văn bản Precision@1 | +5.24 điểm phần trăm |
| SBERT − CLIP, văn bản Recall@5 | +1.97 điểm phần trăm |
| SBERT / CLIP, P50 văn bản | 2.73 lần (SBERT chậm hơn) |

ResNet đã train thấp hơn ResNet pretrained ở toàn bộ chỉ số chất lượng đã đo cho ảnh. Đây là quan sát trên catalog hiện tại, chưa xác định nguyên nhân. Huấn luyện phân loại sáu lớp không bảo đảm tăng chất lượng truy hồi trên 88 danh mục; cần kiểm tra dữ liệu huấn luyện và đánh giá độc lập trước khi quy kết nguyên nhân.

## 10. Đề xuất lựa chọn cho project

| Nhu cầu | Lựa chọn được số liệu hỗ trợ | Đánh đổi |
| --- | --- | --- |
| Tìm ảnh với độ trễ CPU thấp | ResNet18 pretrained | P50 21,041 ms; Precision@1 74,49%; Recall@10 thấp hơn CLIP |
| Tìm mô tả cùng danh mục | SBERT | Precision@1 82,00%; Recall@10 43,52%; P50 102,311 ms |
| Một mô hình hỗ trợ ảnh, văn bản và truy hồi chéo | CLIP pretrained làm mô hình nền | Text→image Recall@10 91,32%; ảnh chậm hơn ResNet trên CPU |
| Chọn checkpoint triển khai CLIP fine-tuned | Chưa kết luận từ lần chạy này | Cần thêm clip_finetuned vào cùng benchmark |
| So với hệ SBERT + ResNet + RRF | Chưa đánh giá | Cần benchmark riêng hệ kết hợp, không suy ra từ hai model độc lập |

Đoạn kết luận đề xuất cho báo cáo: “Trên tập đánh giá gồm 484 sản phẩm, CLIP pretrained đạt Recall@10 41,93% cho truy hồi ảnh cùng danh mục và 91,32% cho truy hồi văn bản sang ảnh đúng sản phẩm. ResNet18 pretrained có ưu thế độ trễ tìm ảnh, còn SBERT có ưu thế chất lượng tìm văn bản cùng danh mục. Với yêu cầu project hỗ trợ đồng thời ảnh và mô tả, nhóm lựa chọn CLIP làm mô hình nền vì cung cấp không gian biểu diễn chung và khả năng truy hồi chéo phương thức. Việc chọn checkpoint fine-tuned cần được xác nhận bằng phép đánh giá tương ứng.”

## 11. Giới hạn và điều kiện sử dụng số liệu

Không xếp hạng SBERT và ResNet bằng cách so trực tiếp điểm của hai tác vụ khác nhau. Không tạo một điểm accuracy tổng hợp từ các bảng này.

Văn bản truy vấn là prompt catalog gồm tên, danh mục và mô tả; danh mục xuất hiện ngay trong đầu vào. Kết quả cùng danh mục là phép đo có hỗ trợ metadata, chưa phản ánh truy vấn tự do của người dùng.

Hai text encoder có giới hạn token khác nhau; benchmark dùng giới hạn mặc định của model. Thời gian và chất lượng chịu ảnh hưởng bởi lượng văn bản thực sự được xử lý.

Danh mục là nhãn thay thế cho mức liên quan, không phải đánh giá phong cách, phối đồ hay hài lòng người dùng.

Chưa xác minh độc lập train/test đối với checkpoint ResNet. Không gọi đây là bằng chứng tổng quát hóa hoàn toàn không rò rỉ dữ liệu.

Một lần chạy CPU và 30 query đo độ trễ chưa đủ suy luận ý nghĩa thống kê hoặc tải đồng thời. Chưa có khoảng tin cậy, độ lệch chuẩn hoặc số đo GPU.

Không đo dịch tiếng Việt, category bonus/hard filtering của ứng dụng, API hay hệ RRF. Không gộp số liệu fine-tuned cũ vào benchmark này.

Không dùng thời gian nạp model hoặc dung lượng tham số làm đại diện cho độ trễ suy luận hoặc RAM thực tế.

## 12. Nguồn và tệp kiểm chứng

Thư mục đầu vào: C:\DACNTT\model\comparison_results\20260929_221746_281836

| File nguồn | SHA-256 |
| --- | --- |
| report.json | 103fbb999d9a69bdb9f129f8a9c93c853348290d185d529ece83e7430c87437e |
| comparison.csv | 9aa29a2ee89ee0b3f0f0a96ff665d2b2ca08e33182fb1add1ca7ff21f43cf91d |
| evaluation_catalog.csv | 50e0259e4fb314352a80bd640817ce2ccc8aa2907e18de4b518ff3d0a8302900 |

audit.json đi kèm lưu các kiểm tra; comparison_percent.csv chứa đầy đủ 21 dòng, Precision/Recall/Hit Rate đã đổi sang %, MRR giữ thang 0–1. Báo cáo này được ghi trong workspace C:\clothes-styling-system\model\reports; dữ liệu nguồn ở C:\DACNTT không bị thay đổi.
