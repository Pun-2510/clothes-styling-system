# Hướng dẫn chạy model và website

Thư mục `model` chứa các model thử nghiệm SBERT, ResNet18, pipeline kết hợp và
website kiểm thử. Các lệnh dưới đây dành cho Windows PowerShell và dùng virtual
environment chung tại `C:\DACNTT\.venv`.

> Chạy từng lệnh riêng và nhấn Enter sau mỗi dòng. Nếu PowerShell hiện dấu
> `>>`, nhấn `Ctrl+C` để hủy câu lệnh chưa hoàn chỉnh rồi chạy lại.

## Cấu trúc thư mục

```text
model/
├── ResNet_Model/       Huấn luyện, audit và checkpoint trung gian ResNet18
├── SBERT_Model/        Huấn luyện và tìm kiếm văn bản bằng SBERT
├── combine_model/      Pipeline kết hợp ảnh + văn bản độc lập
├── Web_Test/           Website kiểm thử và model dùng khi chạy web
├── experiments/        Benchmark, adapter dữ liệu và kết quả thí nghiệm
│   └── results/
│       ├── color/      Kết quả benchmark màu
│       └── comparison/ Kết quả so sánh model
├── docs/screenshots/   Ảnh minh họa giao diện
└── README.md           Hướng dẫn tổng hợp này
```

Không di chuyển bốn thư mục model/web ở trên vì code dùng các đường dẫn tương
đối giữa chúng. Các script benchmark phụ được gom tại `experiments/` để không
lẫn với entry point train và web.

## 1. Cài đặt môi trường

Nếu chưa có virtual environment:

```powershell
Set-Location -LiteralPath 'C:\DACNTT'
py -3.11 -m venv .venv
```

Cài thư viện cho ResNet và website:

```powershell
& 'C:\DACNTT\.venv\Scripts\python.exe' -m pip install -r 'C:\DACNTT\model\ResNet_Model\requirements.txt'
& 'C:\DACNTT\.venv\Scripts\python.exe' -m pip install -r 'C:\DACNTT\model\Web_Test\requirements.txt'
```

Lần chạy đầu cần Internet để tải dataset và model từ Hugging Face. Những lần
sau chương trình ưu tiên cache local.

## 2. Huấn luyện ResNet18 dùng cho website

Entry point huấn luyện là `ResNet_Model/train_resnet.py`. File
`ResNet_Model/train.py` chỉ chứa vòng lặp train và không chạy trực tiếp.

Ví dụ chạy tối đa 10 epoch:

```powershell
Set-Location -LiteralPath 'C:\DACNTT\model\ResNet_Model'
$env:RESNET_EPOCHS = '10'
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\train_resnet.py'
```

Nếu không đặt `RESNET_EPOCHS`, mặc định là 8 epoch. Chương trình tự chọn
`cuda` khi CUDA khả dụng, nếu không sẽ dùng `cpu`. Early stopping dừng sớm khi
macro validation accuracy không cải thiện trong 3 epoch liên tiếp.

Mỗi epoch in dữ liệu theo mẫu:

```text
Epoch [1/10] Loss: 0.4521 Train: 87.25% Val: 91.50% Macro Val: 90.83% LR: 1.00e-04
Saved best model!
```

Để vừa xem trên terminal vừa lưu log:

```powershell
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\train_resnet.py' 2>&1 | Tee-Object -FilePath '.\logs\train_10_epochs.txt'
```

Sau khi train, xóa biến epoch tạm nếu không muốn áp dụng cho lần sau:

```powershell
Remove-Item Env:RESNET_EPOCHS
```

Checkpoint tốt nhất được ghi vào:

```text
C:\DACNTT\model\Web_Test\models\resnet_outfit.pth
C:\DACNTT\model\Web_Test\models\resnet_outfit.json
```

Nếu checkpoint hiện tại có cùng danh sách class, script tự warm-start từ
checkpoint đó. Nên sao lưu trước khi huấn luyện lại:

```powershell
Copy-Item -LiteralPath 'C:\DACNTT\model\Web_Test\models\resnet_outfit.pth' -Destination 'C:\DACNTT\model\Web_Test\models\resnet_outfit_before_train.pth'
```

Muốn bổ sung ảnh thực tế hoặc class mới, đặt ảnh vào
`ResNet_Model/custom_data/<TênClass>/`, khai báo class trong
`ResNet_Model/config.py`, rồi chạy lại `train_resnet.py`. Mỗi class custom cần
ít nhất 20 ảnh.

### Chạy thí nghiệm baseline và pretrained

Lệnh sau huấn luyện các biến thể phục vụ so sánh, không phải entry point chính
để cập nhật checkpoint của website:

```powershell
Set-Location -LiteralPath 'C:\DACNTT\model\ResNet_Model'
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\resnet_model.py'
```

## 3. Kiểm tra model tự động

Chuyển sang đúng thư mục trước khi chạy `app.py`:

```powershell
Set-Location -LiteralPath 'C:\DACNTT\model\Web_Test'
```

Xuất benchmark ResNet18 dưới dạng JSON:

```powershell
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\app.py' --benchmark-resnet
```

Chạy self-test cho text, image và combined search:

```powershell
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\app.py' --self-test
```

`status: pass` xác nhận pipeline tải được model, trả kết quả và sinh score hợp
lệ. Self-test dùng một ảnh catalog để tìm lại trong cùng catalog; nó không thay
thế kiểm thử bằng ảnh upload thực tế.

Chuẩn bị toàn bộ index rồi tự thoát:

```powershell
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\app.py' --warmup combined --prepare-only
```

## 4. Chạy website SBERT + ResNet18

```powershell
Set-Location -LiteralPath 'C:\DACNTT\model\Web_Test'
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\app.py' --warmup combined
```

Mở trình duyệt tại:

```text
http://127.0.0.1:8000
```

`--warmup combined` nạp SBERT và ResNet18 trong nền rồi giữ web server chạy.
Vì vậy terminal không trả lại dấu nhắc PowerShell; đây không phải terminal bị
đơ. Nhấn `Ctrl+C` để dừng server.

Các chế độ warmup khác:

```powershell
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\app.py' --warmup text
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\app.py' --warmup image
```

Nếu cổng 8000 đang được dùng:

```powershell
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\app.py' --warmup combined --port 8001
```

Sau đó mở `http://127.0.0.1:8001`.

## 5. Fine-tune SBERT dùng cho tìm kiếm

Script `train_sbert_retrieval.py` fine-tune SentenceTransformer dùng trực tiếp
cho truy hồi sản phẩm. Mặc định script chạy 3 epoch, lưu model tốt nhất theo
MRR@10 và ghi kết quả từng epoch vào thư mục `logs`.

```powershell
Set-Location -LiteralPath 'C:\DACNTT\model\SBERT_Model'
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\train_sbert_retrieval.py' 2>&1 |
    Tee-Object -FilePath '.\logs\sbert_finetune_terminal.log'
```

Model đầu ra nằm tại `SBERT_Model/models/sbert_fashion_finetuned`. Để website
dùng đúng model vừa train:

```powershell
$env:WEB_SBERT_MODEL = 'C:\DACNTT\model\SBERT_Model\models\sbert_fashion_finetuned'
Set-Location -LiteralPath 'C:\DACNTT\model\Web_Test'
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\app.py' --self-test
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\app.py' --warmup combined
```

`SBERT_Model/train.py` là thí nghiệm BERT classifier cũ, không phải model
SentenceTransformer mà website dùng để tìm kiếm. `predict.py` cũng thuộc luồng
thử nghiệm cũ.

## 6. Chạy riêng pipeline kết hợp

Pipeline này độc lập với website `Web_Test`:

```powershell
Set-Location -LiteralPath 'C:\DACNTT\model\combine_model'
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\main.py'
```

Chạy test của pipeline:

```powershell
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\test.py'
```

## 7. Thứ tự chạy khuyến nghị

1. Cài dependencies vào `C:\DACNTT\.venv`.
2. Huấn luyện ResNet bằng `ResNet_Model/train_resnet.py` nếu cần cập nhật model.
3. Chạy `app.py --benchmark-resnet` và `app.py --self-test`.
4. Chạy `app.py --warmup combined`.
5. Mở `http://127.0.0.1:8000` và kiểm thử thêm bằng ảnh thực tế.

## 8. Benchmark và báo cáo thí nghiệm

Các công cụ không tham gia trực tiếp vào runtime của website nằm tại
`model/experiments`. Xem hướng dẫn chi tiết trong
`model/experiments/README.md`.

Kiểm tra nhanh công thức benchmark, không tải model:

```powershell
Set-Location -LiteralPath 'C:\DACNTT'
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\model\experiments\test_color_benchmark.py'
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\model\experiments\test_mini_fashion.py'
```

Xem cấu hình chi tiết tại:

- `ResNet_Model/README.md`
- `SBERT_Model/README.md`
- `combine_model/README.md`
- `Web_Test/README.md`
- `experiments/README.md`
