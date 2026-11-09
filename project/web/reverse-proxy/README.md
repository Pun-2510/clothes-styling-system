# Web reverse proxy

Đây là public entry point duy nhất của hệ thống:

```text
Browser :80
  ├── /, assets, client routes → frontend:5173
  └── /api, /catalog-images, /docs → api-gateway:80
```

Container này không load model, không rate-limit inference và không trực tiếp
phân phối request tới backend. Các trách nhiệm đó thuộc `api-gateway`.

Kiểm tra cấu hình:

```powershell
docker compose exec reverse-proxy nginx -t
docker compose logs -f reverse-proxy
```
