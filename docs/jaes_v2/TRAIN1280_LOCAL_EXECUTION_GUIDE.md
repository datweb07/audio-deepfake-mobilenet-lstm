# Chạy profile 1.280 trên laptop Windows

**Trạng thái ngày 13/11/2026** — Cohort, kiểm tra tĩnh và smoke 0 update đạt. Bài kiểm tra lifecycle có cập nhật optimizer do tác giả thực hiện. Hai lượt chính chỉ mở sau exact resume, validation parity, export và seal. R03 chưa có kết quả. Hạn manuscript là 20/12/2026.

## 1. Preflight và dung lượng

Mở PowerShell ở repo, dùng venv hiện tại và giữ FFmpeg trên PATH.

```powershell
Set-Location 'D:\audio-deepfake-mobilenet-lstm'
$py = '.\.venv\Scripts\python.exe'
& $py -B -X utf8 -m scripts.jaes_v2.phase10_experiment resources
& $py -B -X utf8 -m scripts.jaes_v2.phase10_size1280 preflight
```

Đọc `qualification_storage.additional_disk_bytes` và `additional_disk_including_qualification_peak_bytes`. Lần đo trong sprint, máy có khoảng 17,18 GiB trống trên D. Profile dành 15 GiB + 128 MiB dự phòng, tối đa 14 MiB mỗi checkpoint, 256 MiB cho export/log/tạm của cặp chính, và ngân sách riêng cho qualification. Trước qualification, mục tiêu cho toàn bộ quy trình là **20,33 GiB trống trên D**. Giải phóng khoảng **3,2 GiB** theo lần đo, sau đó dùng preflight để lấy con số mới. Nếu chỉ chạy qualification, cần khoảng 15,46 GiB trống trên D. Ngưỡng Windows là 2 GiB RAM khả dụng, 3 GiB commit khả dụng và 3 GiB trống C.

Nếu thiếu chỗ, mở Windows Settings → System → Storage hoặc chuyển bản sao tài liệu cá nhân do bạn quản lý. Có thể rà các bản tải trùng và cache build LaTeX thử nghiệm có thể tái tạo. Giữ V6, nguồn LaTeX, `LA.zip`, FLAC, manifest, seal, checkpoint và các báo cáo nghiên cứu. Không có cleanup tự động. Không chạy song song các công việc TensorFlow. Thời gian cặp chính ước tính từ 640 khoảng 5,5 giờ trước dự phòng, nên chừa 5,5–7,5 giờ hoặc hơn và thêm thời gian xác minh. Báo cáo qualification sẽ đưa thêm phép ngoại suy từ bài chạy ngắn.

## 2. Tự chạy qualification

Chạy từng lệnh và kiểm tra `$LASTEXITCODE -eq 0` trước khi chạy lệnh kế. Smoke đã chạy trong sprint và có thể đọc lại bằng chính lệnh dưới đây.

```powershell
& $py -B -X utf8 -m scripts.jaes_v2.phase10_size1280 smoke
& $py -B -X utf8 -m scripts.jaes_v2.phase10_size1280 qualify --role uninterrupted
& $py -B -X utf8 -m scripts.jaes_v2.phase10_size1280 qualify --role resumable
& $py -B -X utf8 -m scripts.jaes_v2.phase10_size1280 qualify-resume --role resumable
& $py -B -X utf8 -m scripts.jaes_v2.phase10_size1280 qualify-verify
& $py -B -X utf8 -m scripts.jaes_v2.phase10_size1280 preflight
```

Lượt đối chiếu chạy 82 update. Lượt còn lại chủ động dừng tại update 80, ngay sau warm-up và validation epoch 1. Xác nhận `PAUSE_AT_CHECKPOINT` và step 80 trước lệnh resume. Hai lượt kết thúc ở update 82 và dùng score grid development đủ 2.576 dòng để kiểm tra restore, export và parity. Nếu gặp lỗi trước checkpoint đầu, giữ thư mục để rà. Nếu lượt uninterrupted bị gián đoạn, giữ bằng chứng và báo lại vì một lượt đã resume không còn là đối chứng uninterrupted. Lượt resumable cần ghi đúng sự kiện resume tại update 80. Các điểm số qualification chỉ dùng kiểm tra kỹ thuật.

Chỉ bắt đầu hai lượt chính khi thấy `SIZE1280_QUALIFICATION_PASS` và preflight ghi `scientific_ready_for_fresh_pair: true`. Nếu số liệu tài nguyên đổi, chạy preflight lại. Lệnh `train` cũng kiểm qualification, khóa chạy tuần tự và đo dung lượng cần cho phần còn lại.

## 3. Train tuần tự và monitor

Terminal A, ERM trước. Xác minh phải trả `SCIENTIFIC_VERIFICATION_PASS` và 2.576 scores trước khi bắt đầu Channel.

```powershell
& $py -B -X utf8 -m scripts.jaes_v2.phase10_size1280 train --arm erm
& $py -B -X utf8 -m scripts.jaes_v2.phase10_size1280 verify --arm erm
& $py -B -X utf8 -m scripts.jaes_v2.phase10_size1280 preflight
& $py -B -X utf8 -m scripts.jaes_v2.phase10_size1280 train --arm channel
& $py -B -X utf8 -m scripts.jaes_v2.phase10_size1280 verify --arm channel
```

Sau ERM, cờ dành riêng cho **cặp mới hoàn toàn** sẽ là false vì ERM đã có output. Lệnh Channel tự xác minh ERM và tính lại phần dung lượng còn cần. Nếu kết quả Channel bất lợi về guard hay AUC, vẫn giữ run và đánh giá như đã đăng ký.

Terminal B, chạy lệnh tương ứng với nhánh đang chạy. Ctrl+C ở terminal này chỉ dừng monitor.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m scripts.jaes_v2.medium_progress --experiment p10_size1280_erm_s9207_v1
.\.venv\Scripts\python.exe -B -X utf8 -m scripts.jaes_v2.medium_progress --experiment p10_size1280_channel_s9207_v1
```

Mỗi nhánh dự kiến đủ 21 epoch và 1.680 update. `train_ETA` là phần optimizer còn lại. Validation và export làm thời gian thực dài hơn.

## 4. Khi bị ngắt

Giữ output, kiểm tra `latest.json` tồn tại và giải quyết nguyên nhân trong `status.json`. Sau khi đo lại tài nguyên, dùng đúng nhánh cần tiếp tục. Lệnh resume xác minh pointer, seal và toàn bộ contract, sau đó restore weights, slots, counter và RNG dựa trên journal. Các update sau checkpoint cuối có thể được tính lại.

```powershell
& $py -B -X utf8 -m scripts.jaes_v2.phase10_size1280 preflight
& $py -B -X utf8 -m scripts.jaes_v2.phase10_size1280 resume --arm erm
# Hoặc dùng dòng sau nếu nhánh bị ngắt là Channel
& $py -B -X utf8 -m scripts.jaes_v2.phase10_size1280 resume --arm channel
```

Với lỗi số học, checkpoint lỗi hash hoặc thiếu pointer, gửi traceback và status để chẩn đoán trước. Giữ nguyên ID, source và cấu hình của run.

## 5. Sau khi đủ hai seal

```powershell
& $py -B -X utf8 -m scripts.jaes_v2.phase10_size1280 compare
```

Lệnh tạo `size_sensitivity.json` và `docs/jaes_v2/TRAINING_SIZE_SENSITIVITY_RESULTS.md`. Nó phát lại số liệu của bốn mô hình trên cùng development, tính Channel−ERM ở mỗi kích thước rồi lấy chênh lệch hai hiệu ứng. Bootstrap dùng cùng trọng số speaker trên bốn mô hình, tách hai stratum. So sánh cố định epoch nên tăng cả số bản ghi và số update, và chỉ có một seed cho contrast kích thước. Giữ V6 cho đến khi các kết quả thực được đưa vào bản V7.

## Bàn giao

- Bây giờ gửi V6, audit R01–R12, response, hướng dẫn này, config `size1280_v1.json` và JSON preflight/integrity/smoke. Bằng chứng 1.280 nằm tại `outputs/jaes_v2/phase10/size1280_v1`. Chọn JSON cụ thể, bỏ `test_dependencies` và `test_scratch*`.
- Sau qualification gửi `qualification/numerical_smoke.json`, `qualification/qualification.json` và `qualification/resources/`.
- Sau train gửi `size_sensitivity.json` và `contract.json`, `result.json`, `scores.csv`, `seal.json` của hai thư mục `scientific/`, kèm hai thư mục `resources/`. Giữ các cây checkpoint và model export để kiểm chứng tại chỗ.
