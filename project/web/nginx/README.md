# Nginx và hai backend

```text
Browser → frontend :5173 → nginx :80 → backend:8000
                                   → backend_2:8000
API trực tiếp localhost:8000 ────→ nginx :80
```

Hai backend cùng hoạt động (active-active). Nginx dùng `least_conn`: ưu tiên
instance có ít kết nối đang hoạt động hơn. Backend không publish cổng ra host;
cổng host 8000 giờ thuộc Nginx. `/api`, `/docs`, `/openapi.json`, `/catalog-images`
vẫn được chuyển tiếp nguyên vẹn. Frontend dùng Vite như trước.

## Chạy

Chạy nhanh từ `project/`: `docker-compose up -d --build`.
Compose tự đọc `.env`; mặc định dùng data/processed và embeddings_finetuned,
không phụ thuộc kết quả comparisons. File Nginx nằm tại web/nginx/default.conf.
Không cần `--env-file` trừ khi muốn ghi đè lựa chọn đó cho một lần chạy.
`.env` là cấu hình riêng từng máy, không lưu Git; xem `.env.example` khi cài mới.

Từ `project/`, khi Docker Desktop đang chạy và đã có model/catalog:

```powershell
docker compose config --quiet
docker compose up -d --build
docker compose exec nginx nginx -t
docker compose ps
docker compose logs -f nginx backend backend_2
```

Chỉ khi muốn xem thử catalog thí nghiệm 4.000 sản phẩm (tùy chọn), thay lệnh `up` bằng:

```powershell
docker compose --env-file comparisons/results/clip_4000_v2/website.env up -d --build
```

Nếu dùng bộ khác với `.env`, giữ `--env-file` khi chạy `up`/tạo lại container,
hoặc cập nhật `.env` một lần. Bỏ `--env-file` sẽ dùng `.env` (nếu có), sau đó mới
đến giá trị mặc định trong Compose. Không cần fine-tune hoặc sinh lại embeddings
chỉ để thêm load balancing. Lần đầu có thể phải tải image Nginx.

## Kiểm tra phân phối request

Đợi model nạp xong rồi gửi nhiều request nhẹ:

```powershell
1..10 | ForEach-Object {
  $response = Invoke-WebRequest -UseBasicParsing http://localhost:8000/api/health/ready
  "HTTP $($response.StatusCode) | upstream=$($response.Headers['X-Upstream-Addr'])"
}
```

Header `X-Upstream-Addr` và log Nginx cho biết IP backend thực sự được gọi. Hai
IP khác nhau cho thấy request đã đi qua hai instance; không yêu cầu tỉ lệ luôn
50/50. Header phục vụ debug local, nên bỏ khi public nếu không muốn công khai
IP nội bộ. Response sau retry có thể ghi cả hai IP.

## Thử dự phòng — tự chạy khi chấp nhận tạm dừng một instance

```powershell
docker compose stop backend
Invoke-RestMethod http://localhost:8000/api/health/ready
# Thử tìm kiếm trên website: backend_2 phải còn hoạt động và đã nạp model.
docker compose start backend
docker compose ps
```

Đợi `backend` healthy trở lại rồi có thể thử tương tự với `backend_2`.
Không dừng cả hai cùng lúc. Không dùng `down` để thử lỗi một instance vì nó
dừng toàn bộ hệ thống. `start` dùng lại container đã có, không đổi catalog.

## Hành vi và giới hạn

- Nginx phát hiện lỗi thụ động qua request, không chủ động thăm dò readiness.
  Sau một lỗi kết nối/timeout/502/503/504, peer bị tạm tránh trong 10 giây;
  mỗi request thử tối đa hai lần. Docker healthcheck hiển thị trạng thái riêng,
  không tự xóa peer khỏi upstream hay restart chỉ vì container unhealthy.
- Gateway chờ hai container được start, không chờ cả hai healthy. Web có thể
  báo chưa sẵn sàng lúc model đang nạp. Một model lỗi khởi tạo không ngăn
  gateway phục vụ qua instance còn lại.
- `/lb-health` chỉ kiểm tra Nginx sống. `/api/health/ready` kiểm tra backend
  nhận request, không khẳng định cả hai sẵn sàng. Readiness hiện có thể trả 200
  khi `translator_ready=false`; kiểm tra trường này nếu dùng tiếng Việt.
  API text trả 503 khi bộ dịch không sẵn sàng và có thể thử peer còn lại.
- Retry POST chỉ bật ở hai endpoint inference `/api/recommendations/image`
  và `/api/recommendations/text` vì không ghi dữ liệu nghiệp vụ. Timeout có
  thể làm tính toán/log lặp; không cam kết thực thi đúng một lần. Không bật
  retry kiểu này cho endpoint lưu dữ liệu/thanh toán trong tương lai.
- Connect timeout 3 giây; chờ dữ liệu response 120 giây. Retry chỉ được bắt
  đầu trong cửa sổ 130 giây, không phải giới hạn tổng thời gian end-to-end.
  Không đảm bảo chuyển ngay lập tức hay cứu response đã gửi một phần.
  Không retry lỗi đầu vào 4xx hoặc lỗi code 500.
- Body tối đa 6 MiB ở Nginx để chứa ảnh 5 MiB cùng multipart. API vẫn kiểm
  tra ảnh tối đa 5 MiB, MIME và các tham số như trước.
- Docker DNS được kiểm tra lại định kỳ để nhận IP mới khi container được
  tạo lại. `resolve` yêu cầu Nginx >= 1.27.3; dùng `nginx:stable-alpine`.
- Hai instance chia sẻ artifact chỉ đọc và volume cache tải model Hugging
  Face nhưng model/embeddings trong RAM là riêng. Cần đủ RAM và đo tải thật;
  không mặc định throughput tăng gấp đôi khi chia sẻ CPU. Khi đổi artifacts,
  tạo lại cả hai backend; không ghi đè model/CSV/embeddings đang được dùng.
- Log ứng dụng: `logs/recommendation-backend.log` và
  `logs/recommendation-backend_2.log`. Nginx ghi upstream/timing ra Docker logs.
- Chưa có cache truy vấn/Redis. Cache RAM nếu thêm sau này sẽ riêng cho mỗi
  backend; muốn chia sẻ thì cân nhắc Redis trong một thay đổi riêng.
- Dự phòng này chỉ ở mức tiến trình trên cùng máy. Nginx, Docker host và
  artifact chung vẫn là điểm lỗi chung, không phải triển khai HA nhiều máy.

Tài liệu: [upstream/least_conn/resolve](https://nginx.org/en/docs/http/ngx_http_upstream_module.html),
[quy tắc retry](https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_next_upstream).

Kiểm tra cấu hình không thay thế thử tải hoặc failover thực tế. Chỉ ghi nhận
failover thành công sau khi chạy các bước trên và lưu kết quả/log.
