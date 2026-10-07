# Thử nghiệm tag màu sắc trên ResNet, SBERT, CLIP và Combine

`color_benchmark.py` là thử nghiệm riêng, không đổi trọng số, benchmark cũ hoặc API của web. Chạy lệnh từ thư mục gốc repository. Python 3.11+; dùng môi trường đã cài torch, torchvision, transformers, sentence-transformers, NumPy, pandas và Pillow như `compare_models.py`.

Dataset mặc định là **Mini Fashion của project**, cấu hình dùng chung qua [MINI_FASHION.md](MINI_FASHION.md). Sau khi chọn project, có thể bỏ `--csv`, `--image-dir`, `--clip-checkpoint`. Lần chạy `color_tags_pilot` đã dùng chính Mini Fashion này.

## Mục tiêu và cách chấm điểm

Đo khả năng tìm **sản phẩm khác cùng danh mục và có màu yêu cầu**. Đây là đánh giá truy hồi theo màu, không phải accuracy của bộ phân loại màu và chưa phải khả năng phối đồ.

- Đầu vào CSV: `product_id,product_name,category,image_path`, tùy chọn `description`; cần `split` nếu dùng mặc định `--split test`. Một dòng cho mỗi product ID, không ảnh trùng byte.
- Nhãn ban đầu lấy từ **tên sản phẩm**, không lấy từ model đang đánh giá hay mô tả dài (mô tả có thể nói về phụ kiện khác).
- Nhãn này là **weak label chưa kiểm chứng bằng ảnh**. Tên thương hiệu, ánh sáng và sản phẩm nhiều màu có thể gây sai; không coi đây là ground truth.
- Dùng 14 nhóm màu rộng: `black,white,gray,blue,red,green,yellow,orange,pink,purple,brown,beige,gold,silver`. Navy → blue, maroon → red, grey → gray; quy tắc đầy đủ nằm ở `ALIASES`. Không phân biệt sắc độ.
- Nhiều màu lưu bằng `|`, ví dụ `black|pink`. Query yêu cầu một màu; sản phẩm có chứa màu đó được tính khớp màu.
- Mỗi sản phẩm sinh một query. Nếu có nhiều màu, chọn một màu cố định bằng hash product ID. Query văn bản là tiếng Anh, ví dụ `black Tshirts`; query ảnh là ảnh sản phẩm đó.
- Loại chính product ID khỏi toàn bộ gallery. Chỉ chấm query có sản phẩm khác cùng danh mục và màu; ghi số query bỏ qua. Tất cả model/biến thể dùng cùng cohort, query và thứ tự catalog đã trộn bằng seed 42.
- Precision/Recall/Hit Rate/MRR chấm theo **cả danh mục và màu**; `color_precision` chỉ chấm màu. `comparison.csv` dùng thang 0–1; Markdown/HTML đổi các tỷ lệ sang % (MRR giữ 0–1).
- Precision chia cho `min(K, số ứng viên trước lọc)`: nếu bộ lọc trả ít hơn K thì không tự nâng điểm vì giảm mẫu số.

## Các model và phép thử

| Tên CLI | Đầu vào | Cách tìm |
| --- | --- | --- |
| `resnet18` | Ảnh | ResNet pretrained, cosine embedding |
| `resnet18_trained` | Ảnh | ResNet checkpoint hiện có |
| `sbert` | `color category` | Query text → văn bản catalog |
| `clip` | Ảnh hoặc text | Báo riêng ảnh→ảnh, text→text, text→ảnh |
| `clip_finetuned` | Ảnh hoặc text | Cùng tác vụ với CLIP, dùng checkpoint fine-tune |
| `combine` | Ảnh + text | ResNet pretrained + SBERT, RRF |
| `combine_trained` | Ảnh + text | ResNet đã train + SBERT, RRF |

Combine áp dụng `1/(60+rank)` cho mỗi nhánh, trọng số bằng nhau, trên **toàn bộ ranking ứng viên**, gộp theo identity sản phẩm. Công thức tương ứng `combine_model/test.py` và `Web_Test/app.py`. Đây là benchmark có kiểm soát trên cùng dataset; không chạy nguyên trạng ứng dụng cũ vốn có cách nạp dữ liệu/tiền xử lý riêng. `combine_model/combine_search.py` hiện trả hai danh sách riêng, không tự hợp nhất bằng RRF.

| Biến thể | Thay đổi |
| --- | --- |
| `baseline` | Embedding và ranking gốc, không lọc tag |
| `text_tags` | Thêm `Color: black. ...` vào đầu văn bản catalog rồi encode lại; dùng cho nhánh text SBERT/CLIP và SBERT của Combine |
| `tag_filter` | Giữ scores baseline, chỉ trả sản phẩm có tag màu yêu cầu |

ResNet và CLIP text→ảnh không có văn bản catalog để thêm tag nên không có `text_tags`. Ảnh không bị tô/chèn chữ tag. Query không thay đổi giữa các biến thể.

Không xếp hạng tất cả model như thể chúng nhận cùng lượng thông tin: ResNet baseline chỉ thấy ảnh (chưa nhận tên màu yêu cầu), SBERT chỉ thấy text, Combine thấy cả hai. Lọc tag bổ sung thông tin màu rõ ràng cho nhánh ảnh; điều này đặc biệt quan trọng với sản phẩm nhiều màu.

**Không diễn giải tag_filter là model thông minh hơn:** nhãn được dùng cả để lọc và chấm điểm, nên độ đúng màu tăng là kết quả mong đợi của bộ lọc metadata. Cần xem baseline, kiểm chứng nhãn độc lập và đo cả cùng danh mục. Văn bản baseline vốn có thể đã chứa màu trong tên; `text_tags` đo việc đưa tag rõ ràng lên đầu, không phải so sánh có/không có mọi thông tin màu.

## Chạy thử

Ví dụ dùng dữ liệu và môi trường đã có ở `C:\DACNTT`, nhưng lưu kết quả tại repository đang mở:

```powershell
& C:/DACNTT/model/.venv/Scripts/python.exe -B model/color_benchmark.py `
  --csv C:/DACNTT/project/runs/clip_finetune_3epochs_local/dataset.csv `
  --image-dir C:/DACNTT/project/data/data `
  --resnet-checkpoint C:/DACNTT/model/Web_Test/models/resnet_outfit.pth `
  --clip-checkpoint C:/DACNTT/project/runs/clip_finetune_3epochs_local/best `
  --device cpu
```

Mặc định chạy cả 7 cấu hình ở K=1,5,10, batch=16. `--output model/color_results/ten_lan_chay` chọn thư mục **chưa tồn tại**; nếu bỏ qua sẽ tự sinh timestamp. Có thể giảm phạm vi bằng `--models resnet18 sbert clip combine` khi chưa có checkpoint đã train. `--device cuda` chỉ dùng khi có GPU/PyTorch hỗ trợ.

Lần đầu cần tải model pretrained, hoặc dùng cache có sẵn. Để bắt buộc dùng cache offline, đặt `HF_HUB_OFFLINE=1` và `TRANSFORMERS_OFFLINE=1`. Không đổi CSV thí nghiệm cũ; `--image-dir` định vị lại ảnh bằng tên file.

## Kiểm chứng nhãn trước khi báo cáo chính thức

1. Chạy lệnh trên với thêm `--prepare-only`; bước này chỉ tạo catalog, query và tài liệu review, không nạp model.
2. Mở `color_review.html` để đối chiếu tên sản phẩm với ảnh. Sửa `color_tags` trong `color_labels_review.csv`, dùng tên chuẩn và dấu `|`; đặt `reviewed=true` cho dòng đã kiểm tra. Không đặt true hàng loạt nếu chưa xem ảnh.
3. Chạy lại với `--labels <đường-dẫn-color_labels_review.csv>` và thư mục output mới. Chỉ các dòng reviewed=true được đưa vào cả gallery và query. Có thể thêm ID chưa tự trích được màu nếu ID đó có trong split nguồn.
4. Nếu cohort thay đổi, không so trực tiếp điểm tổng với lần chạy cũ. Dùng baseline và các biến thể trong **cùng lần chạy nhãn đã review**.

Template review ban đầu chỉ chứa những sản phẩm trích được nhãn; các sản phẩm không có màu trong tên bị loại và được đếm trong `excluded_rows`. Với nhiều mẫu ảnh không xác định được màu, để chưa review thay vì đoán.

## Kết quả và báo cáo

| File | Nội dung |
| --- | --- |
| `color_catalog.csv` | Catalog, tag, nguồn nhãn, hash ảnh và văn bản |
| `color_labels_review.csv`, `color_review.html` | File sửa nhãn và trang đối chiếu ảnh |
| `queries.csv` | Query cố định dùng chung cho tất cả model |
| `report.json` | Cấu hình, nguồn dữ liệu/hash, độ phủ nhãn, thời gian encoding, trạng thái/lỗi |
| `comparison.csv`, `summary.md` | Bảng số liệu và giới hạn phép đo |
| `per_query.csv` | Điểm mỗi query, ID sản phẩm trả về để phân tích lỗi |

Tạo HTML và bảng chênh lệch từ một lần chạy đã xong, không cần nạp model:

```powershell
python model/build_color_report.py --source model/color_results/color_tags_pilot
```

Mở `color_report.html`; `deltas.csv` chứa chênh lệch điểm phần trăm so với baseline của đúng model/tác vụ/K. Các file raw có đường dẫn máy và ảnh review được ignore; báo cáo tổng hợp có thể đưa lên Git.

Đọc `status` và `errors` trong JSON. `partial_failure` nghĩa là có model chưa chạy được, không coi bảng là phép so sánh đầy đủ. Chạy lại vào thư mục mới sau khi sửa lỗi. Script này không tương thích mẫu `build_comparison_report.py` của benchmark cũ.

Kiểm tra công thức, không tải model:

```powershell
& C:/DACNTT/model/.venv/Scripts/python.exe -B model/test_color_benchmark.py
```

Thời gian `encode_seconds` là tạo embedding, không phải thời gian API/query. Chưa kiểm chứng overlap tập train của các checkpoint. Thử nghiệm chưa đánh giá query tiếng Việt, màu thực tế do ánh sáng hay chất lượng phối đồ.
