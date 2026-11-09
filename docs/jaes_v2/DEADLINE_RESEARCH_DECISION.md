# Quyết định nghiên cứu đến 20/12/2026

`TRAINING_ENGINE_VERSION = FROZEN_PHASE10`. Cặp prospective ERM và channel seed 9107 đã seal, cùng source order ở cả 840 update và cùng contract ngoài augmentation. Tăng stress AUC 0,6715 → 0,8026 và giảm stress FAKE miss 222/336 → 155/336 trên A. Đồng thời clean B false alarm tăng 19/144 → 31/144. Guard trên hai checkpoint được chọn trượt A clean miss, B clean false alarm và A06 stress miss. Kết quả chỉ mô tả một cặp seed trên development. [Audit](../../outputs/jaes_v2/phase10/deadline_s9107_audit_v1.json)

| Đóng góp | Trạng thái | Phạm vi có thể bảo vệ |
| --- | --- | --- |
| C1 | Supporting empirical evidence | Đối chứng matched và lỗi cùng source dưới thứ tự channel, với kiểm toán theo lớp và attack. Khác phạm vi conference, vẫn cần replication và held-out trước khi mở rộng claim. |
| C2 | `NOVELTY_NOT_ESTABLISHED` | A có tín hiệu đáng kiểm tra bằng paired diagnostics, nhưng objective đơn giản quy về consistency hoặc worst-view. B bị loại ở sprint hiện tại vì chưa có cue độc lập kiểm chứng được. |
| C3 | Deferred | Chi phí CPU và tính tái lập phục vụ báo cáo kỹ thuật. Abstention hoặc cascade chuẩn chưa tạo khác biệt phương pháp. |

Ưu tiên MobileNetV3Small–LSTM128 vì đã có engine CPU và contract matched được xác minh. ShuffleNetV2 chỉ dùng để kiểm tra transfer khi có C2 đủ bằng chứng. AASIST là pretrained reference nếu audit được nguồn checkpoint và overlap. RawNet2 bổ sung góc nhìn waveform với cùng yêu cầu provenance. MnasNet và EfficientNet-B0 giữ vai trò kết quả conference.

Chốt matrix tối thiểu cho hướng empirical là ba cặp ERM/channel medium. Seed 9107 hoàn tất. Seed 9207 rồi 9307 chạy tuần tự khi tài nguyên đạt, với cùng config đã đăng ký. Không đưa seed 6107 vào replication. C2 hiện chưa có objective được duyệt. Medium-plus và foundation giữ NO-GO vì chưa có ứng viên hoặc câu hỏi scaling đã đạt gate. [Protocol](DEADLINE_NOVELTY_AND_EXPERIMENT_PROTOCOL.md) · [Lệnh](DEADLINE_TRAINING_COMMANDS.md)

| Mốc | Sản phẩm và điều kiện |
| --- | --- |
| 09–12/11 | Xác minh cặp 9107, khóa endpoints và xem nguồn lực cho cặp tiếp theo. |
| 13–17/11 | Tối đa một vòng kiểm tra cơ chế order có đối chứng, đồng thời viết manuscript. Chỉ pilot training nếu có technical distinction. |
| 18–20/11 | Quyết định cuối về C2. Nếu cơ chế hoặc đối chứng gần nhất vẫn giải thích tín hiệu, khóa paper hướng empirical. |
| 21–30/11 | Hoàn tất các seed chính nếu tài nguyên cho phép. Giữ cùng protocol, báo cáo cả trường hợp guard thất bại. |
| 01–07/12 | Đánh giá held-out đúng bản đăng ký, chỉ khi hash và provenance đã khóa. |
| 08–14/12 | Phân tích, bảng/figure, uncertainty và rà claim. |
| 15–20/12 | QA, disclosure extension, venue/copyright và gói reproducibility. |

Rủi ro chính là thiếu system commit khi bắt đầu run, chi phí CPU và khoảng cách giữa evidence development với claim journal. Bản thảo có thể tiến triển ngay. Ba novelty quốc tế chưa khả thi để cam kết ở mốc này.
