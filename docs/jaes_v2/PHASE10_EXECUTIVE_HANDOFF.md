# Phase 10 — Bàn giao cho tác giả

## Kiểm tra lúc 22:48 sau khi system commit phục hồi

Preflight đạt với C 8,03 GiB, D 24,38 GiB, RAM khả dụng 4,33 GiB và commit khả dụng 7,36 GiB. Backend cũ `SEGMENTWISE_VJP` đạt 7/9 fixture. Ở clip REAL noise5, đổi shape batch đẩy một preactivation qua ngưỡng đạo hàm hard-swish −3. Backend v2 `FULL_SHAPE_RECOMPUTE` giữ nguyên shape khi fine-tune và đạt 9/9 theo dung sai ban đầu. [Tài nguyên](../../outputs/jaes_v2/phase10/resources_ram_recheck_v1.json) · [Gradient v1](../../outputs/jaes_v2/phase10/numeric_parity.json) · [Gradient v2](../../outputs/jaes_v2/phase10/numeric_parity_v2.json)

## Mẫu lịch sử sau khi tác giả giải phóng ổ C

Ổ C đã phục hồi lên khoảng 6,71 GiB. Phase 10 CLI giờ giới hạn OpenBLAS và OMP ở một luồng trước khi nạp NumPy. Process private commit đo được giảm từ khoảng 782 MiB xuống 76 MiB. Một mẫu sau thay đổi đạt 2,10 GiB RAM khả dụng nhưng chỉ 2,77 GiB commit, dưới ngưỡng 3 GiB. Một lần thử gradient sau đó vượt preflight, nạp TensorFlow và tạo model, rồi dừng ở kiểm tra commit trong vòng fixture. Chưa có báo cáo gradient hoàn chỉnh, exact resume, full-cohort inference parity hay benchmark 100 update. **NO-GO vẫn áp dụng cho cả ba quy mô train.** [Mẫu đo sau thay đổi](../../outputs/jaes_v2/phase10/resources_recovery_pinned_v2.json)

## Kết quả Phase 10 và các gate

| Mục | Quyết định và bằng chứng |
| --- | --- |
| A. Engine đã thống nhất chưa? | `TRAINING_ENGINE_FROZEN=YES` cho CPU medium với `FULL_SHAPE_RECOMPUTE`. Launcher khoa học khóa release và hai seal pilot. |
| B. Gradient parity? | **PASS v2**, 9/9 fixture. Kết quả v1 trượt 2 ca partial fine-tuning được giữ làm bằng chứng. |
| C. Exact resume? | **PASS v2** ở lifecycle 9 update, resume từ update 5. Tất cả 17 phép so sánh logic đạt. File weights bên trong archive trùng byte. Archive khác timestamp và ID đối tượng Keras. |
| D. Full validation? | **PASS v2** trên checkpoint ERM đã seal, 2.576 cặp score, 368 source. Max 6,26e-6, mean 1,96e-7, 0 flip. AUC, PR-AUC, EER và group counts trùng. |
| E. Benchmark? | `ENGINE_100_UPDATE_PASS`. Hai lượt hoàn tất 100 update, comparator resume ở 55, 17 phép so sánh logic đạt. Lượt liên tục ~11 phút 39 giây update, ~14 phút 52 giây validation, ~724 MiB peak RSS. |
| F. Tài nguyên? | Mẫu 22:48 qua gate. TensorFlow 2.15.0 CPU chạy hết fixture, peak RSS khoảng 804 MiB. |
| G. Medium replication? | **GO có preflight tại thời điểm chạy.** Bắt đầu với ERM seed 9107, rồi channel cùng seed. |
| H. Medium-plus? | **NO-GO.** Có manifest 2.560 train. Cần prospective matched pair và lý do mở rộng. |
| I. Foundation? | **NO-GO.** 25.380 train có preflight Phase 9. Chi phí thực tế và giá trị khoa học chưa đủ. |
| J. C2 novelty? | `C2_CUE_IDENTIFICATION_BLOCKED_ON_DATA` và `C2_NOVELTY_NOT_ESTABLISHED`. Đã chốt phép kiểm chứng cue theo source và thứ tự transform. |
| K. Cần dataset mới ngay? | Chưa. Dataset hiện có đủ cho engineering. Mechanism C2 cần parallel lineage được xác minh trước khi thu thập mới. |
| L. Chạy gì? | Preflight rồi ERM seed 9107, ngắt có kiểm soát ở update 40. [Gói lệnh](PHASE10_AUTHOR_TRAINING_COMMANDS.md) |

Ưu tiên MobileNetV3Small–LSTM128. Giữ 6107 như bằng chứng thăm dò. Dự kiến ERM/channel với seed mới 9107, 9207, 9307 khi đủ điều kiện. Nếu đến 20/11 vẫn thiếu dữ liệu cue hoặc engine qualification, thu hẹp mục tiêu tới evidence development. Không đặt kế hoạch nộp 20/12 dựa trên ba đóng góp chưa được xác nhận. [Master report](LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT_PHASE10.md)
