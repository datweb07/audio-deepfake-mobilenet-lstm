# Quyết định huấn luyện sau held-out

**11/11/2026. `NO_ADDITIONAL_TRAINING_REQUIRED`.** Medium seed 9307, medium-plus, full foundation và train lại sáu detector đều **NO-GO** cho bản JAES hiện tại. Full foundation mang trạng thái `FULL_TRAINING_NOT_SCIENTIFICALLY_NECESSARY`. Mọi training mới vẫn `NOT_AUTHORIZED`.

## Bằng chứng và phạm vi kết luận

Bốn run MobileNetV3Small–LSTM128 giữ 640 nguồn, 21 epoch và 840 update mỗi arm. Hai cặp ERM/Channel dùng seed 9107 và 9207 với cùng quy trình chọn checkpoint. Các seal và fingerprint engine khớp. Held-out đã khóa có 1.088 nguồn, 67 speaker, 256 REAL, 832 FAKE từ A07–A19 và 17.408 dự đoán. Audit đọc manifest, file score, shards, metrics và hash. Số dưới đây lấy từ artifact đã xác minh.

| Endpoint held-out, ERM → Channel | Seed 9107 | Seed 9207 |
| --- | ---: | ---: |
| Macro stress ROC-AUC | 0,8308 → 0,8875 | 0,7498 → 0,8948 |
| Stress FAKE miss | 1.370/2.496 → 587/2.496 | 815/2.496 → 453/2.496 |
| Stress REAL false alarm | 80/768 → 81/768 | 322/768 → 136/768 |
| Clean ROC-AUC | 0,9534 → 0,9436 | 0,9507 → 0,9410 |
| Clean REAL false alarm | 25/256 → 19/256 | 22/256 → 16/256 |
| Clean FAKE miss | 125/832 → 129/832 | 127/832 → 147/832 |
| A17 / A18 / A19 stress miss tăng | +10 / +11 / +4 mỗi 192 view | +43 / +43 / +56 mỗi 192 view |

Macro stress AUC tăng +5,67 và +14,50 điểm phần trăm. Khoảng 95% speaker-cluster trong từng seed là [+2,96; +8,42] và [+11,58; +17,52]. Stress miss giảm −31,37 và −14,50 điểm phần trăm. A17–A19 xấu hơn ở cả hai seed, với khoảng 9107 bao gồm 0 và khoảng 9207 nằm trên 0. Hai seed chia sẻ evaluation cohort. Các khoảng mô tả bất định qua speaker cho từng cặp. Ba view stress cùng một nguồn và các contrast theo attack là phân tích nhóm nhỏ.

Ba đóng góp của V2 có dữ liệu tương ứng. Matched comparison kiểm tra sự thay đổi do chính sách augmentation trong thiết kế medium. Lỗi lớp và attack cho thấy trade-off dù AUC trung bình cải thiện. Score và transition cùng nguồn xác lập tính nhạy theo thứ tự của hai chuỗi âm học đã thử. Các kết luận giới hạn ở một kiến trúc, hai initialization và một benchmark release. Cơ chế forensic độc lập, độ ổn định trên quần thể seed, transfer sang kiến trúc khác, cross-corpus và deploy thực tế vẫn là câu hỏi mở. [Bản thảo](../../papers/jaes_v2/jaes_manuscript/build/LAVA_JAES_Review_Draft_v2.pdf) · [Held-out](../../outputs/jaes_v2/phase10/empirical_execution_v1/heldout_28f940ea7d151177/analysis/HELDOUT_SCIENTIFIC_REPORT.md)

## Bảy câu hỏi về full foundation

| Câu hỏi | Kết luận |
| --- | --- |
| Full giải đáp điều gì mới? | Có thể kiểm tra tương tác giữa quy mô train và hiệu ứng channel. Câu hỏi đó khác với câu hỏi medium đã đăng ký và cần tách số nguồn khỏi tỷ lệ REAL/FAKE. |
| Có giả thuyết thực chất? | Một giả thuyết khả dĩ là tăng số nguồn làm đổi stress benefit và worst-attack risk dưới cùng weighting. Một run nhắm AUC cao hơn không ước lượng được tương tác. Hiện chưa có bằng chứng độc lập để đầu tư. |
| Đối chứng công bằng? | Về thiết kế có thể dùng MobileNetV3Small–LSTM128 và cặp ERM/Channel cùng seed, cohort, optimizer, quota, update budget và selection. So sánh scale cần thêm medium matched hoặc factorial size × class-mix, với estimand theo exposure hoặc update được khóa trước. Chưa có hợp đồng full như vậy. |
| Có xác nhận độc lập cho model mới? | Chưa có tập xác nhận mới đủ provenance, license và lineage audit được khóa cho câu hỏi này. Official evaluation hiện tại đã được xem. Chọn model hoặc hparam từ nó rồi chấm lại chỉ là exploratory. |
| Kịp 20/12? | Chưa có bằng chứng về khả năng hoàn thành profile, storage, matched runs, xác nhận độc lập, phân tích và phản biện. Với retention hiện tại ổ D còn thiếu ngay cả cho một arm theo kịch bản. |
| Bắt buộc để mở rộng conference? | Hướng dẫn Research Paper của AES nhấn mạnh đóng góp và quy trình có thể tái lập. Audit không thấy yêu cầu dùng toàn bộ ASVspoof. Việc chấp nhận còn phụ thuộc novelty thực nghiệm, nội dung mở rộng và biên tập. [AES Research Paper](https://aes.org/publications/aes-journal-manuscript-types/) |
| Có củng cố ba đóng góp hiện có? | Scale study có thể mở rộng scope, nhưng bốn run và held-out đã trực tiếp hỗ trợ ba đóng góp hẹp. Information gain dự kiến của full trước deadline thấp hơn chi phí và nguy cơ đổi câu hỏi. |

Thêm một cặp medium giúp có thêm initialization nhưng lợi ích hiện tại là hẹp, vẫn cùng kiến trúc và cohort. Giữ quyết định bỏ seed 9307 và mô tả rõ hai block. Dùng checkpoint legacy để đối chiếu lịch sử hoặc inference có protocol riêng. Không gộp bảng kết quả conference và V2 thành một bảng xếp hạng. C2 vẫn rejected và C3 deferred. [Ma trận legacy](LEGACY_SIX_MODEL_REUSE_MATRIX.md) · [Feasibility](FULL_FOUNDATION_FEASIBILITY_20261111.md)

## Tiêu chí thành công cho một nghiên cứu tương lai

Đây là điều kiện để xem xét đề xuất mới, không phải protocol được phê duyệt. Trước khi huấn luyện, khóa giả thuyết, population, matched comparator, seed, cohort, estimand, threshold, kết quả chính, uncertainty, subgroup support, budget và stop rules trên grouped development folds độc lập với held-out đã mở.

| Tiêu chí | Điều kiện đăng ký trước |
| --- | --- |
| Clean ROC-AUC | Đặt margin non-inferiority từ precision trên speaker folds và mức giảm có ý nghĩa sử dụng. Báo từng seed, paired interval và support. Chưa có margin được duyệt cho full. |
| Clean REAL false alarms và clean FAKE misses | Với đúng nhóm dev đã đăng ký, guard hiện có cho A clean FA, A clean miss và B clean FA cho phép tối đa +2 điểm phần trăm so với reference đồng mẫu. A clean FA cap 10%, A clean miss cap 15%. Mọi thay đổi population hoặc threshold cần hiệu chuẩn trên dev trước khi khóa. |
| Stressed macro ROC-AUC | Khóa ba stress conditions, hướng lợi ích, minimum meaningful effect và cách bootstrap speaker. Yêu cầu cải thiện nhất quán trong các block theo rule đã định, cùng với guard ở lớp và attack. Không lấy chênh lệch held-out hiện tại làm target mới. |
| Stressed REAL false alarms và FAKE misses | Dùng class-specific denominators và báo cả hai. Guarded selection hiện dùng mean của hai stress risks nếu qua các guard. Cần cap suy giảm cho từng lớp từ cost/tolerance và support trên dev. Một endpoint tốt lên không bù tự động endpoint kia. |
| Worst attack | A05/A06 guard hiện cho phép +5 điểm phần trăm trên cùng cohort. Với protocol mới, khóa nhóm, tối thiểu support, adverse margin và quy tắc xử lý uncertainty trước khi xem kết quả. Thiếu denominator hoặc vượt guard thì reject. |
| Class imbalance | Đăng ký natural hoặc balanced exposure và weights cho cả hai arm. Báo số source/label/attack, số lần tái sử dụng, effective batch và step budget. Tách scale khỏi class-mix hoặc công bố rõ estimand hỗn hợp. |
| Stability và reproducibility | Stop khi nonfinite, gradient zero/exploding, hợp đồng resume lệch hoặc thiếu score. Kiểm tra exact model/optimizer/cursor và export trong isolated qualification. Báo từng block và sự cố, không chọn bỏ run bất lợi. |
| Chi phí | Khóa trần wall-clock, peak RAM/commit, disk/retention và ngân sách ít nhất cho cặp đối chứng. Nghiệm thu bằng workload train-only trước khi cam kết. |

Hiện còn thiếu mức cải thiện tối thiểu có căn cứ sử dụng, margin cho các endpoint mới, F3 và tập xác nhận độc lập. Vì vậy chưa thể khóa tiêu chí full đáng tin. Kết quả tương lai không thể bảo đảm. [Guard hiện hành](../../src/lava/jaes_v2/phase10_validation.py) · [Protocol thực nghiệm](EMPIRICAL_EXECUTION_PROTOCOL.md)

## Monitor và bàn giao

Monitor Phase 10 vẫn là tiến trình đọc riêng. Nó dùng registry đã kiểm và contract nếu có, hiển thị epoch, update bar, phần trăm, loss/LR/norm, validation AUC/BCE/counts, checkpoint, resource warning và resume event. Elapsed là thời gian mở monitor. ETA chỉ tính optimizer. Dữ liệu thiếu hiện N/A. Status cũ chỉ báo rằng cần kiểm tra terminal trainer vì schema không có PID để xác nhận trainer đã thoát. Wrapper một terminal chưa có hợp đồng an toàn về Ctrl+C và vòng đời process group trên Windows. Vì vậy giữ monitor đọc riêng. Cả 6 display legacy dùng đường riêng.

Chạy từ root để xem một run đã hoàn tất mà không khởi động trainer:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m scripts.jaes_v2.medium_progress --experiment p10_medium_channel_s9207_v2 --once
.\.venv\Scripts\python.exe -B -X utf8 -m scripts.jaes_v2.medium_progress --experiment p10_medium_erm_s9107_v2 --once
.\.venv\Scripts\python.exe -B -X utf8 -m scripts.jaes_v2.phase10_experiment resources
```

Ví dụ rút gọn từ run ERM đã seal:

```text
Epoch 21/21
Optimizer updates [████████████████████████] 840/840 — 100.0%
Last completed validation: tier B at update 840
Clean ROC-AUC 0.9886798469387755 | Clean BCE 0.1564222850396791
Clean REAL false alarms 8/112 | FAKE misses 5/112
Best checkpoint u000560_e014_c000 | Latest u000840_e021_c000
System status SCIENTIFIC_COMPLETE | Resume: Recorded at update 180
```

**Kiểm thử trong lượt audit.** 31 pytest fixture/static checks và 15 subtest đạt. Năm unittest tổng hợp cho objective, denominator, checkpoint restore và lỗi atomic write đạt trong scientific venv. Lượt pytest đầu qua Python hệ thống gặp lỗi import TensorFlow/Keras, sau đó các case liên quan được chạy bằng venv đã khóa. Một selected model được restore, kiểm hash weights và chạy forward trên tensor zero `(1, 6, 224, 224, 3)` để xác nhận score hữu hạn. Toàn bộ seal bốn medium run, 2.376 mục freeze legacy với ba amendment được đăng ký, và 28 mục inventory Phase 9 đều khớp. Preflight của release medium hiện hành cũng đạt. Hai demo monitor giữ nguyên hash các tệp đã đọc. Audit tái tính metrics và bootstrap từ score đã lưu, với 17.408 prediction và 1.088 shard, khớp output gốc. [Integrity report](../../outputs/jaes_v2/phase10/post_heldout_integrity_20261111.json)

Khi gửi AI độc lập, dùng báo cáo này, ma trận sáu model, feasibility, JSON audit và PDF V2. Kèm held-out metrics, manifest, verified analysis và scores CSV nếu cần tính lại. Thêm monitor và test fixture nếu muốn rà code. Checkpoints và audio nguồn có thể lưu tại máy. Trước 17/11 chốt quyết định. Phần thời gian còn lại dành cho góp ý giảng viên, làm rõ contribution so với conference, reference/rights và QA nguồn LaTeX. [JSON quyết định](../../outputs/jaes_v2/post_heldout_training_audit_v1.json) · [Handoff V2](JAES_HELDOUT_V2_HANDOFF.md)
