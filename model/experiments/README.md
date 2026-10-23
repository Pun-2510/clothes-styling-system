# Benchmark và thí nghiệm model

Thư mục này chứa các công cụ đánh giá độc lập, không tham gia trực tiếp vào
quá trình chạy website.

## Thành phần

| File/thư mục | Mục đích |
| --- | --- |
| `compare_models.py` | So sánh ResNet18, SBERT và CLIP trên cùng catalog |
| `color_benchmark.py` | Đánh giá truy hồi có thuộc tính màu |
| `build_comparison_report.py` | Tạo báo cáo từ kết quả so sánh model |
| `build_color_report.py` | Tạo HTML và bảng delta cho benchmark màu |
| `mini_fashion.py` | Adapter đọc dataset Mini Fashion local |
| `test_*.py` | Unit test không cần tải model |
| `results/color/` | Kết quả benchmark màu đã lưu |
| `results/comparison/` | Báo cáo so sánh model đã lưu |

## Kiểm tra nhanh

Chạy từ thư mục gốc repository:

```powershell
Set-Location -LiteralPath 'C:\DACNTT'
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\model\experiments\test_color_benchmark.py'
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\model\experiments\test_mini_fashion.py'
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\model\experiments\compare_models.py' --self-test
```

## Chạy benchmark màu

```powershell
& 'C:\DACNTT\.venv\Scripts\python.exe' '.\model\experiments\color_benchmark.py' --prepare-only
```

Nếu không truyền `--output`, kết quả mới được tạo trong
`model/experiments/results/color/<timestamp>`.

Xem thêm:

- [COLOR_BENCHMARK.md](COLOR_BENCHMARK.md)
- [MINI_FASHION.md](MINI_FASHION.md)
