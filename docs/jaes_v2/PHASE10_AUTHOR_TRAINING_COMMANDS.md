# Phase 10 — Huấn luyện đối chứng medium

**GO cho medium trên CPU khi preflight tại thời điểm chạy đạt.** Mỗi cặp khóa cùng MobileNetV3Small–LSTM128, 640 train, 368 development, thứ tự source, initialization, optimizer, backend, 840 update và `CLEAN_BCE`. Khác biệt trong cặp là `clean` so với `channel`. [Đăng ký sáu run](../../configs/jaes_v2/phase10/prospective_medium_v1.json)

Chạy từ root repo trong PowerShell. Bắt đầu ERM seed 9107. Hai preflight seed 9107 đã chạy trên launcher này. Cơ chế train, checkpoint, resume, xuất và đối chiếu đã chạy trong pilot cùng engine. Các run khoa học dành cho tác giả.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium preflight --experiment p10_medium_erm_s9107_v2
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium train --experiment p10_medium_erm_s9107_v2 --stop-at 40
```

Lệnh thứ hai dự kiến chấm clean A ở cuối epoch 1, commit update 40 và dừng an toàn. Trong terminal khác có thể theo dõi trạng thái. Chỉ resume sau khi đã có checkpoint commit.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium monitor --experiment p10_medium_erm_s9107_v2
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium resume --experiment p10_medium_erm_s9107_v2
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium verify --experiment p10_medium_erm_s9107_v2
```

Khi ERM đã seal, chạy channel cùng seed rồi đối chiếu hai run. Có thể thêm `--stop-at 40` nếu muốn chủ động ngắt lần đầu. Dùng đường dẫn JSON mới để lưu kết quả.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium preflight --experiment p10_medium_channel_s9107_v2
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium train --experiment p10_medium_channel_s9107_v2
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium verify --experiment p10_medium_channel_s9107_v2
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase10_medium compare --experiment p10_medium_erm_s9107_v2 --peer p10_medium_channel_s9107_v2 --output outputs/jaes_v2/phase10/medium_pair_s9107_v2.json
```

Run output nằm dưới `outputs/jaes_v2/phase10/<experiment>/scientific`. Sau commit có `latest.json`, `status.json` và checkpoint đã hash. Run hoàn tất có `selected.keras`, `scores.csv`, `result.json` và `seal.json`. So sánh đọc hai seal và kiểm tra cấu hình ghép cặp trước khi tính guard. Giữ kết quả dù trượt guard. Lặp tuần tự cho seed 9207 và 9307 bằng ID trong registry. Medium-plus và foundation có quyết định riêng.

Dừng tại checkpoint nếu guard báo thiếu RAM, commit hoặc dung lượng. Giữ C ≥3 GiB, D ≥15 GiB +128 MiB, RAM khả dụng ≥2 GiB và commit ≥3 GiB. Khi có `STOP_FOR_NUMERICAL_FAILURE` hoặc hash sai, kiểm tra evidence trước khi thử lại. [Hướng dẫn tài nguyên](PHASE10_WINDOWS_STORAGE_ACTION.md)
