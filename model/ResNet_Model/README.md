# Fashion Classification using ResNet18

## Dataset

Dataset được tải từ Hugging Face:

ashraq/fashion-product-images-small

## Cài đặt

```bash
python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

# Chạy huấn luyện Baseline và Pretrained ResNet18
python resnet_model.py

## Huấn luyện thêm class sườn xám

Checkpoint hiện tại chỉ có 6 class và không thể tự nhận ra một class chưa từng
được học. Đặt tối thiểu 20 ảnh vào `custom_data/Cheongsam/`, sau đó chạy:

```powershell
Set-Location C:\DACNTT\model\ResNet_Model
..\.venv\Scripts\python.exe train_resnet.py
```

Script dùng balanced sampling để class custom không bị lấn át, chọn checkpoint
có validation accuracy tốt nhất và tự ghi kết quả vào
`../Web_Test/models/resnet_outfit.pth`. Khởi động lại Web_Test sau khi train.

## Phân loại và xử lý ảnh truy vấn

`classification.py` chứa riêng logic phân loại loại trang phục và trả về nhãn,
confidence cùng xác suất của từng lớp. Luồng tìm kiếm trong `combine_model`
xóa nền ảnh người dùng bằng `rembg`, ghép vật thể lên nền trắng, rồi dùng cùng
ảnh đã xử lý cho cả classification và trích xuất embedding.
