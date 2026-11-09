# Lệnh sprint cho tác giả

Cặp seed 9107 đã hoàn tất. Lệnh compare đã chạy và lưu [JSON](../../outputs/jaes_v2/phase10/medium_pair_s9107_v2.json). `resume` chỉ dùng cho run chưa seal. Hai run 9107 hiện có final seal, vì vậy không cần chạy lại.

## Seed 9207

Chạy từ repository root. Kiểm tra tài nguyên khi máy ở trạng thái rảnh. Lần audit ngày 09/11 chặn ở `SYSTEM_COMMIT_PRESSURE`, khoảng 2,09 GiB commit và 1,89 GiB RAM khả dụng trong mẫu gần preflight. Gate là commit ≥3 GiB, RAM khả dụng ≥2 GiB, C ≥3 GiB, D ≥15 GiB +128 MiB. Lưu công việc, đóng ứng dụng nặng đang rảnh rồi đo lại.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium preflight --experiment p10_medium_erm_s9207_v2
```

Khi báo `MEDIUM_PREFLIGHT_PASS` và chưa có thư mục run:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium train --experiment p10_medium_erm_s9207_v2
```

Theo dõi trong terminal khác. Nếu chủ động chia đoạn, đặt `--stop-at` theo số update tuyệt đối chưa tới. Chỉ resume từ checkpoint đã commit. Nếu báo lỗi số học hoặc hash sai, lưu evidence để kiểm tra trước khi tiếp tục.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium monitor --experiment p10_medium_erm_s9207_v2
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium resume --experiment p10_medium_erm_s9207_v2
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium verify --experiment p10_medium_erm_s9207_v2
```

Sau khi ERM seal, chạy channel tuần tự, rồi so sánh. Không chạy hai jobs nặng cùng lúc.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium preflight --experiment p10_medium_channel_s9207_v2
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium train --experiment p10_medium_channel_s9207_v2
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium verify --experiment p10_medium_channel_s9207_v2
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium compare --experiment p10_medium_erm_s9207_v2 --peer p10_medium_channel_s9207_v2 --output outputs/jaes_v2/phase10/medium_pair_s9207_v2.json
```

Launcher và workflow đã được kiểm chứng bằng pilot và cặp 9107. Seed 9207 chưa train trong sprint này. Dùng IDs đã đăng ký cho 9307 sau khi có cặp 9207 hợp lệ. [Registry](../../configs/jaes_v2/phase10/prospective_medium_v1.json)

Output mỗi run nằm trong `outputs/jaes_v2/phase10/<experiment>/scientific`. Khi gửi AI sau hoàn tất, dùng `result.json`, `scores.csv`, `contract.json`, `seal.json` cho mỗi run, cùng JSON compare. Checkpoint payload chỉ cần nếu bên nhận sẽ restore.
