# LAVA Journal V2 — Master research report sau hai lượt medium

08/11/2026, Asia/Novosibirsk. Phiên bản này bổ sung kết quả của hai lượt medium đã seal. Bản Phase 8 lịch sử có SHA-256 `633572c15f3acd8df50d393e1691c4b905bb7222d165344144a9d5370cc39773` và giữ toàn bộ bảng lịch sử, baseline hội nghị, danh mục dữ liệu và evidence Phase 1–8 tại thời điểm bàn giao trước training. [Master Phase 8](LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT_PHASE8.md) · [Inventory Phase 8](LAVA_JOURNAL_V2_RESEARCH_INVENTORY_PHASE8.json)

## Tiến trình

| Phase | Evidence đã chạy | Giới hạn chính |
| --- | --- | --- |
| 1 | Ba pilot thứ tự noise/Opus trên nguồn ghép và bootstrap theo parent. | Noise insertion có thể gây nhiễu giải thích order. |
| 2 | Fixed/relative noise, headroom, PCM và delay control. | Cần tách mức nhiễu, gain và codec. |
| 3 | MLAAD-tiny development với hai artifact frozen. | Tên hệ thống chưa xác minh unseen checkpoint. |
| 4 | Audit source matching và nuisance, MobileNet và ShuffleNet. | Lineage thiếu và số cặp đúng generator không đủ cho xác nhận. |
| 5 | Foundation ASV và pilot head-only P0/P1/P2, calibration được đặt riêng. | Gain invariance có thể đi kèm giảm phân biệt lớp. |
| 6 | B0/B1/B2/B4, equal-view control, finite RawBoost và sanity partial. | Cohort và budget khác Phase 5, kết quả cơ chế chưa được thiết lập. |
| 7 | Cohort all-attack có A/B, ba seed cho baseline frozen và F0/F1 matched 40 update. | Consistency H3−H2 có mean stress miss delta −0.00397, SD 0.08174, cả ba block trượt guardrail. F1 clean AUC tăng từ 0.7508 đến 0.8104, stress miss tăng. |
| 8 trước medium | M1 weighted BCE và M2 group-DRO, 18 lượt tiny gồm 17 lượt đánh giá và một ca infeasible, trainer resume và audit. | Tiny smoke kiểm tra đường chạy, không xác nhận novelty. M3 không vượt gate. |
| 8 medium | ERM và channel partial seed 6107, cùng 640 train và 368 development, 840 update/21 epoch mỗi run, mỗi model 2.576 score. | Một seed, development dùng lặp, trainer migration khác nhau và không có frozen đối chứng 840 update. |

## Kết quả quyết định

| Metric | ERM | Channel |
| --- | --- | --- |
| Epoch chọn, clean A BCE | 17, 0.12535 | 16, 0.16297 |
| Clean A FA, miss | 4/112, 8/112 | 4/112, 8/112 |
| Clean A ROC-AUC, EER | 0.98868, 5.36% | 0.97911, 5.36% |
| Stress A FA, miss | 137/336, 162/336 | 72/336, 128/336 |
| Stress A macro ROC-AUC, EER | 0.62112, 41.82% | 0.80841, 26.79% |
| B clean FA | 19/144 | 30/144 |
| B stress FA | 122/432 | 93/432 |
| A05 stress miss | 40/54 | 47/54 |
| A06 stress miss | 19/54 | 37/54 |

Stress gồm noise5 và hai thứ tự noise/codec, macro theo condition. Score ở threshold 0.5 và danh sách đầy đủ denominators, per-attack, per-speaker và flips nằm trong review. Hai model đều cập nhật 6/206 tensor backbone từ warmup tới export chọn, 0/136 tensor BatchNorm đổi. Channel dùng đường tính gradient theo segment từ step 426 và pointwise từ 430. AUC tốt hơn dưới stress nhưng chi phí theo lớp/attack vẫn tồn tại. Chênh lệch chỉ là mô tả của một cặp model được chọn. [Review](PHASE8_MEDIUM_TRAINING_SCIENTIFIC_REVIEW.md) · [Audit và decision JSON](PHASE8_POST_TRAINING_DECISION.json)

## Đóng góp và những gì chưa kiểm chứng

| Ứng viên | Trạng thái sau medium | Điều kiện để tiến tiếp |
| --- | --- | --- |
| C1 — ảnh hưởng order và lỗi theo lớp | Bằng chứng thực nghiệm và engineering trên cohort đã dùng. | Xác nhận ghép nguồn, kiểm soát nuisance và nguồn độc lập có lineage. |
| C2 — detector nhẹ với bằng chứng robust | NOVELTY_NOT_ESTABLISHED. Weighted BCE, group-DRO, ERM và channel là đối chứng. | Phép đo cue phản thực nghiệm cần qua falsification và matched controls. |
| C3 — policy inference thích ứng/abstention | Deferred. | Calibration, risk–coverage, chi phí và cohort độc lập được khóa trước. |

Chưa xác nhận kết quả nhiều seed của partial medium, khả năng chuyển sang generator hoặc corpus độc lập, calibration, official evaluation, GPU và tổng chi phí train so sánh. Việc kiểm tra 840 update chỉ xác nhận hai baseline development có thể hoàn tất. Quyết định là **YES** cho replication hẹp có protocol khóa trước, **NO** cho foundation và train C2 mới. [Hướng Phase 9](PHASE9_NOVELTY_RESEARCH_DIRECTION.md)

## Lệnh đã kiểm tra

Các lệnh đọc evidence dưới đây dùng được trên checkout hiện tại. Audit dựng lại metric và kiểm hash mà không chạy model hoặc huấn luyện. CLI help dùng để kiểm tra cú pháp. Không có lệnh full training được kích hoạt trong vòng review.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_medium_review
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.phase8_experiment --help
```

Khi có protocol replication đã đăng ký, dùng ID mới cho config thay đổi và preflight trước khi chạy. Lệnh train cho config tương lai chờ config và quyết định tài nguyên tương ứng.
