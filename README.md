# Category-Aware Multimodal Fashion Retrieval using CLIP

Hệ thống truy xuất sản phẩm thời trang trong một catalog bằng **ảnh**, **văn bản** hoặc **cả hai**. Người dùng nhận danh sách Top-K sản phẩm tương tự; đây là công cụ tìm kiếm sản phẩm, không phải hệ thống phối đồ hay bán hàng.

Mô hình CLIP được fine-tune trên các cặp ảnh–mô tả thời trang. Ảnh và văn bản được mã hóa thành embedding, sau đó so sánh với sản phẩm trong catalog bằng cosine similarity. TF-IDF bổ sung mức khớp từ khóa giữa truy vấn và tên, mô tả sản phẩm. Hệ thống hiện dùng **Hard Category** để xác định tập ứng viên cho truy vấn ảnh và truy vấn kết hợp.

## Quy trình hoạt động

**Ngoại tuyến:** kiểm tra dữ liệu và ảnh → lấy mẫu catalog, thống kê phân bố danh mục → chia train/validation/test → fine-tune CLIP → tạo embedding ảnh và văn bản → xây chỉ mục TF-IDF → chọn trọng số truy xuất trên validation. Các artifact vận hành được tạo và lưu trong `project/`; website không lấy checkpoint, embedding hay trọng số từ `experiments/`.

![Quy trình chuẩn bị dữ liệu, mô hình và artifact ngoại tuyến](docs/reports/figures/figure-offline-pipeline.png)

**Trực tuyến:** giao diện gửi ảnh, văn bản hoặc cả hai → Nginx chuyển yêu cầu tới FastAPI → hệ thống mã hóa truy vấn, tính điểm và chọn tập ứng viên → trả Top-K. Ba luồng được xử lý như sau:

![Online Retrieval Pipeline: image, text and combined queries](docs/reports/figures/figure-online-retrieval-pipeline.png)

| Truy vấn | Cách xếp hạng hiện tại |
| --- | --- |
| Chỉ ảnh | So sánh ảnh truy vấn với ảnh catalog; dự đoán một danh mục từ bằng chứng thị giác và CLIP category prompts; xếp hạng các sản phẩm trong danh mục đó. |
| Chỉ văn bản | Kết hợp mức khớp text→image và text→text với TF-IDF của tên, mô tả. Chỉ giới hạn vào một danh mục khi truy vấn nêu rõ đúng một danh mục. |
| Ảnh + văn bản | Kết hợp điểm ảnh và văn bản. Danh mục nêu rõ trong văn bản được ưu tiên; nếu không có, dùng danh mục dự đoán từ ảnh. TF-IDF tinh chỉnh thứ hạng trong danh mục đã chọn. |

Ảnh có thể được crop thủ công để tập trung vào sản phẩm cần tìm. Bộ lọc đối tượng `Nữ / Nam / Trẻ em / Unisex` là tùy chọn riêng do người dùng chọn; hệ thống không suy đoán giới tính từ ảnh. Website còn có duyệt danh mục, chi tiết sản phẩm, Favorites, lịch sử tìm kiếm trên thiết bị và trang Developer.

## Category-Aware Ranking đã phát triển

Ban đầu, sản phẩm được xếp hạng theo độ tương đồng mà không xét danh mục. Dự án đã thử các cách xử lý danh mục để giảm việc một loại sản phẩm khác chen vào Top-K:

1. **Hard filter:** chỉ xếp hạng sản phẩm thuộc danh mục được chọn hoặc dự đoán.
2. **Soft support:** cộng một mức hỗ trợ cố định cho sản phẩm cùng danh mục, vẫn giữ các danh mục khác trong tập ứng viên.
3. **Confidence-weighted support:** điều chỉnh mức hỗ trợ theo độ tin cậy của dự đoán danh mục.
4. **Dynamic relation-aware support:** xét thêm quan hệ giữa các danh mục gần nhau và phân bổ mức hỗ trợ tùy tình huống.

**Bản website hiện triển khai hard filter** cho ảnh và truy vấn kết hợp. Danh mục ảnh được dự đoán bằng hai nguồn: các sản phẩm tương tự nhất được lấy cân bằng theo từng danh mục và độ khớp giữa ảnh truy vấn với các câu CLIP prompt mô tả danh mục. Cách này đạt kết quả cao nhất trong thí nghiệm dùng *cùng danh mục* làm tiêu chí đúng. Hạn chế của nó là nếu dự đoán sai danh mục, sản phẩm phù hợp ở danh mục khác sẽ bị loại; người dùng có thể crop lại ảnh hoặc thêm mô tả nêu rõ loại sản phẩm. Các cách soft và dynamic được giữ trong hồ sơ thí nghiệm để giải thích quá trình phát triển, không phải lựa chọn trên website hiện tại.

## Cấu trúc và cách chạy

| Thư mục | Vai trò |
| --- | --- |
| [`project/`](project/) | Pipeline chuẩn bị dữ liệu, huấn luyện, artifact vận hành và website. Có thể đóng gói riêng thành `source/`. |
| [`experiments/`](experiments/) | Script và kết quả so sánh các mô hình, chiến lược xếp hạng; không phải đầu vào của website. |
| [`docs/reports/evidence/final_experiments/`](docs/reports/evidence/final_experiments/) | Bảng kết quả và hồ sơ chạy cuối cùng. |

Đọc [hướng dẫn chạy trong project/README.txt](project/README.txt) để biết yêu cầu phần mềm, cấu trúc dataset và từng lệnh chuẩn bị catalog, fine-tune, sinh embedding, chọn trọng số và khởi động Docker. Tên thư mục sau khi clone có thể thay đổi; các lệnh trong hướng dẫn được chạy từ bên trong `project/` hoặc bản nộp `source/`.

Khi đã có đủ artifact và `.env`, khởi động từ thư mục `project/`:

```powershell
docker compose config --quiet
docker compose up -d --build
```

Website ở `http://localhost`, Swagger ở `http://localhost/docs`, Developer Dashboard ở `http://localhost/developer`.

Chi tiết thí nghiệm và cách đọc bảng kết quả nằm tại [experiments/model_comparison/README.md](experiments/model_comparison/README.md) và [hồ sơ kết quả cuối](docs/reports/evidence/final_experiments/README.md). Các baseline chỉ dùng model và raw cosine được trình bày riêng với pipeline hoàn chỉnh vì pipeline còn áp dụng lọc danh mục và TF-IDF.
