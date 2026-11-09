# Phase 10 — Khôi phục dung lượng và bộ nhớ trên Windows

Đo ngày 08/11/2026 theo giờ Novosibirsk. Mẫu đầu từ PowerShell cho thấy C còn 7,19 GiB. Sau đó Windows tăng commit limit từ 22,65 lên 26,69 GiB, trong khi C giảm còn khoảng 3,15 GiB. Các mẫu đo chỉ cho thấy biến động đồng thời. Mẫu sau regression ghi nhận D 21,70 GiB, RAM 15,32 GiB tổng và 1,22 GiB khả dụng, commit còn 2,54 GiB. Mẫu audit sớm hơn chỉ còn 0,90 GiB commit. Windows registry báo pagefile hiện có trên C. WMI/CIM bị từ chối trong sandbox nên chưa đo được kích thước và lượng pagefile sử dụng trực tiếp. Native Win32 API đo system commit, physical RAM và process RSS. [Mẫu sớm](../../outputs/jaes_v2/phase10/resources_v1.json) · [Mẫu cuối](../../outputs/jaes_v2/phase10/resources_v3.json)

## Điều kiện thử lại

Mẫu 22:48 đạt gate với C 8,03 GiB, D 24,38 GiB, RAM khả dụng 4,33 GiB và commit khả dụng 7,36 GiB. Snapshot do tác giả gửi trước đó có RAM khả dụng 1,83 GiB, thấp hơn ngưỡng 2 GiB. Trong nhánh đó, `PROCESS_RSS_PRESSURE` là mã hiện tại cho thiếu RAM khả dụng dù RSS của CLI chỉ khoảng 97 MiB. Khi mẫu mới giữ đủ tài nguyên, công việc tiếp theo nằm ở nghiệm thu backend. [Mẫu đạt](../../outputs/jaes_v2/phase10/resources_ram_recheck_v1.json)

Lần kiểm tra lại sau khi tác giả dọn ổ C ghi nhận khoảng 6,71 GiB trống. Với OpenBLAS/OMP giới hạn một luồng trong CLI, mẫu kiểm tra có 2,10 GiB RAM khả dụng và 2,77 GiB commit, thiếu khoảng 0,23 GiB trước khi tính phần TensorFlow cần lúc khởi tạo. Ưu tiên đóng ứng dụng nặng đang rảnh để giải phóng thêm 1–2 GiB và lấy lại hai mẫu đo. Gate hiện bị chặn bởi system commit. [Mẫu kiểm tra lại](../../outputs/jaes_v2/phase10/resources_recovery_pinned_v2.json)

Gate kỹ thuật yêu cầu C ≥3 GiB, D ≥15 GiB +128 MiB dự phòng ghi, commit còn ≥3 GiB, RAM khả dụng ≥2 GiB, process RSS ≤4 GiB. Dung lượng đĩa, RAM và commit được đo riêng.

1. Lưu công việc. Trong Task Manager, xem Memory và Committed ở tab Performance. Tự đóng các ứng dụng người dùng đã lưu và đang rảnh.
2. Mở Settings → System → Storage → Cleanup recommendations hoặc Temporary files. Xem từng mục trước khi xác nhận. Disk Cleanup cũng có thể kiểm tra dung lượng file tạm trên C. [Microsoft Support](https://support.microsoft.com/en-us/windows/experience/storage-filemanagement/storage-settings-in-windows)
3. Giữ nguyên pagefile, registry, system files, Downloads chưa kiểm tra, môi trường Python, dataset archives, checkpoint, lịch sử Git và kết quả nghiên cứu.
4. Chạy lại lệnh resources trong [gói audit](PHASE10_AUTHOR_TRAINING_COMMANDS.md). Chụp ít nhất hai mẫu khi máy đã ổn định. Nếu gate vẫn fail, giữ trạng thái RESOURCE_BLOCKED.

CLI kiểm tra khả năng tạo file tạm rồi đặt TEMP/TMP riêng cho tiến trình qualification trên D trong thư mục Phase 10. Lần thử mới đã khởi tạo TensorFlow với thiết lập này, rồi dừng do commit trong vòng fixture. CUDA driver thấy RTX 3050 Laptop 4 GiB. Phase 10 dùng contract CPU và GPU TensorFlow vẫn chưa được xác minh.
