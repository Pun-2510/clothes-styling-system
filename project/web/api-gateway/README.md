# Internal API Gateway

```text
Web Reverse Proxy :80
        ↓ internal /api, /catalog-images, /docs
API Gateway :80
        ├── least_conn → backend:8000
        └── least_conn → backend_2:8000
```

API Gateway không publish port ra host. Nó chịu trách nhiệm:

- Rate limiting theo IP.
- Load balancing `least_conn`.
- Bounded retry khi backend lỗi hoặc timeout.
- Giới hạn body/timeout cho inference.
- Gắn `X-Upstream-Addr` để kiểm tra backend phục vụ request.

Tất cả request từ trình duyệt đi vào Reverse Proxy trước, vì vậy website,
Swagger và API cùng dùng origin `http://localhost`.

## Kiểm tra

```powershell
docker compose exec api-gateway nginx -t
docker compose logs -f api-gateway backend backend_2
Invoke-RestMethod http://localhost/api/health/ready
```

Kiểm tra phân phối request:

```powershell
1..10 | ForEach-Object {
  $response = Invoke-WebRequest -UseBasicParsing http://localhost/api/health/ready
  "HTTP $($response.StatusCode) | upstream=$($response.Headers['X-Upstream-Addr'])"
}
```

Hai địa chỉ upstream khác nhau chứng minh cả hai backend đều nhận request;
`least_conn` không đảm bảo phân phối luôn đúng 50/50.

## Giới hạn

- Hai backend cùng chia sẻ artifact trên đĩa nhưng mỗi process giữ model và
  embeddings riêng trong RAM.
- Retry POST chỉ bật cho image-only và text-only inference hiện không ghi dữ
  liệu nghiệp vụ. Không áp dụng tùy tiện cho endpoint có side effect.
- API Gateway và Reverse Proxy vẫn chạy trên cùng Docker host, nên đây không
  phải high availability nhiều máy.
