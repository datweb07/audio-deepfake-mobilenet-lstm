# Tổng hợp khoa học sau seed 9207

10/11/2026. Phạm vi là development ASVspoof 2019 LA.

## Kiểm toán

Bốn run p10_medium_erm_s9107_v2, p10_medium_channel_s9107_v2, p10_medium_erm_s9207_v2 và p10_medium_channel_s9207_v2 có seal hợp lệ, đủ 840 updates và 21 epoch. Mỗi run có 368 bản ghi development và 2.576 score theo lưới nguồn–điều kiện. Kiểm tra hash file model xuất, checkpoint được chọn, controller, score và contract. Initialization và thứ tự sample của 840 updates khớp trong từng cặp. Backend FULL_SHAPE_RECOMPUTE và CLEAN_BCE cùng giữ cố định. ID Phase 8 trong config lồng là metadata của template, còn ID thực nằm ở trường experiment của contract. Audit lịch sử kiểm tra lại baseline freeze, inventory Phase 9 và hai seal Phase 8. Tệp conference khớp SHA-256 91fa8753025895bbf88470f4ebe7e6c283e897c8370fa8e91b1719abaac61073.

Nguồn đối chiếu gồm empirical execution protocol, final scientific diagnostic, C2 decision, deadline reports, Phase 10 master report, bốn run contracts, hai pair comparisons, Phase 10 engine/validation/resources và các script audit/held-out. Kết quả số dưới đây được tái tính từ score gốc. A05/A06 của 9207 lệch bảng tham chiếu trong đề bài đúng một lỗi mỗi arm. Giá trị có seal là 38→47 và 16→31, với 54 view mỗi ô.

## Kết quả development

| Development endpoint | 9107 ERM | 9107 Channel | 9207 ERM | 9207 Channel |
| --- | ---: | ---: | ---: | ---: |
| Selected update | 560 | 760 | 520 | 720 |
| A clean ROC-AUC | 0.9872 | 0.9814 | 0.9851 | 0.9809 |
| A clean EER | 4.46% | 4.91% | 5.36% | 5.36% |
| A macro stressed ROC-AUC | 0.6715 | 0.8026 | 0.5946 | 0.7839 |
| A clean REAL false alarms | 3/112 | 3/112 | 4/112 | 4/112 |
| A clean FAKE misses | 6/112 | 9/112 | 9/112 | 9/112 |
| A stressed REAL false alarms | 54/336 | 45/336 | 167/336 | 70/336 |
| A stressed FAKE misses | 222/336 | 155/336 | 143/336 | 134/336 |
| B clean REAL false alarms | 19/144 | 31/144 | 26/144 | 20/144 |
| A01 stressed FAKE misses | 21/57 | 2/57 | 13/57 | 1/57 |
| A02 stressed FAKE misses | 47/57 | 46/57 | 27/57 | 45/57 |
| A03 stressed FAKE misses | 38/57 | 8/57 | 30/57 | 7/57 |
| A04 stressed FAKE misses | 27/57 | 4/57 | 19/57 | 3/57 |
| A05 stressed FAKE misses | 48/54 | 50/54 | 38/54 | 47/54 |
| A06 stressed FAKE misses | 41/54 | 45/54 | 16/54 | 31/54 |

| Channel − ERM | 9107 [95% cluster CI] | 9207 [95% cluster CI] | Mean | Range |
| --- | ---: | ---: | ---: | ---: |
| Stress ROC-AUC | +0.1312 [+0.0207, +0.2387] | +0.1893 [+0.0932, +0.2684] | +0.1602 | +0.1312 to +0.1893 |
| A clean REAL false alarms (pp) | +0.00 [-2.68, +2.68] | +0.00 [-2.68, +2.68] | +0.00 | +0.00 to +0.00 |
| A clean FAKE misses (pp) | +2.68 [+0.00, +6.25] | +0.00 [-3.57, +4.46] | +1.34 | +0.00 to +2.68 |
| A stressed REAL false alarms (pp) | -2.68 [-8.63, +4.76] | -28.87 [-37.20, -19.05] | -15.77 | -28.87 to -2.68 |
| A stressed FAKE misses (pp) | -19.94 [-25.31, -14.58] | -2.68 [-10.42, +3.57] | -11.31 | -19.94 to -2.68 |
| B clean REAL false alarms (pp) | +8.33 [+1.39, +17.36] | -4.17 [-6.94, -0.69] | +2.08 | -4.17 to +8.33 |
| A01 stressed FAKE misses (pp) | -33.33 [-40.00, -25.93] | -21.05 [-29.82, -11.67] | -27.19 | -33.33 to -21.05 |
| A02 stressed FAKE misses (pp) | -1.75 [-13.73, +10.00] | +31.58 [+22.22, +40.35] | +14.91 | -1.75 to +31.58 |
| A03 stressed FAKE misses (pp) | -52.63 [-71.93, -33.33] | -40.35 [-61.67, -20.00] | -46.49 | -52.63 to -40.35 |
| A04 stressed FAKE misses (pp) | -40.35 [-56.25, -25.00] | -28.07 [-49.02, -7.94] | -34.21 | -40.35 to -28.07 |
| A05 stressed FAKE misses (pp) | +3.70 [-3.70, +12.70] | +16.67 [+3.70, +29.82] | +10.19 | +3.70 to +16.67 |
| A06 stressed FAKE misses (pp) | +7.41 [-6.67, +22.92] | +27.78 [+11.76, +43.75] | +17.59 | +7.41 to +27.78 |

Both seed blocks show higher aggregate stressed ranking (+0.1312 and +0.1893 AUC units) and fewer A stressed FAKE misses (−19.94 and −2.68 percentage points). The 9207 miss interval crosses zero. A clean REAL false-alarm counts match within each block, A clean FAKE misses increase in 9107 and tie in 9207, and clean B REAL false alarms reverse direction (+8.33 and −4.17 points). A05 and A06 worsen in both seed blocks. Their 9107 intervals include zero. A02 also worsens in 9207. The registered selected-checkpoint comparison fails A clean miss, B clean false alarm and A06 stress miss in 9107, and A05/A06 stress miss in 9207. These guards describe the selected CLEAN_BCE checkpoints.

Mỗi CI percentile dùng 10.000 lượt bootstrap theo cụm speaker với seed 91109. A và B lấy mẫu riêng, giữ nguyên mọi nguồn và view của cùng speaker. Tệp JSON lưu số lượt hợp lệ/không hợp lệ, ước lượng theo speaker, các transition ghép nguồn và hash đầu vào. Các khoảng phản ánh biến thiên speaker có điều kiện trên model đã fit. Hai initialization dùng chung cohort, nên mean/range chỉ mô tả hai block. Mỗi attack có support nhỏ, các view lặp phụ thuộc và các phân tích phụ không điều chỉnh đa so sánh.

## A05/A06 và ảnh hưởng thứ tự

| Seed / attack | Clean | Noise5 | Noise→codec | Codec→noise | Missed in all three | Harmed / recovered sources |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 9107 / A05 | 0→0 | 17→17 | 14→16 | 17→17 | 14→16 | 3 / 1 |
| 9107 / A06 | 3→6 | 18→16 | 7→12 | 16→17 | 6→12 | 7 / 2 |
| 9207 / A05 | 0→0 | 17→17 | 7→13 | 14→17 | 7→13 | 8 / 0 |
| 9207 / A06 | 5→7 | 9→12 | 0→5 | 7→14 | 0→5 | 10 / 0 |

Each row contains 18 original recordings and 54 stressed views. Each condition has a denominator of 18. Harmed and recovered denote at least one cross-model threshold crossing on a stress view and may overlap. A05 clean errors were zero throughout. In both seeds its excess stress misses came from ordered processing, with Noise5 unchanged. A06 already incurred clean errors, and its largest excess came from Noise→codec in 9107 and Codec→noise in 9207. At 9207, 8/18 A05 and 10/18 A06 sources were harmed, with no recovered sources for those two attacks. At 9107, the counts were 3/18 and 7/18, with some recovery. Source-level transitions and per-condition mean FAKE scores are in the numerical artifact. These are fixed-threshold transitions on FAKE-only subsets. Attack-only ROC-AUC is undefined and no generator mechanism can be inferred. A05 and A06 were represented during training.

Among 112 A REAL sources, Noise→codec alone caused 43→19 errors in 9107 and 63→31 in 9207, with zero reverse-only errors. Among 112 A FAKE sources, Codec→noise alone caused 48→17 and 40→26 errors, respectively. Noise→codec-only FAKE errors were 1→0 and 0→0. Both models can shift scores in the same direction, producing different decision effects by class at 0.5. Only 9107 has matched standalone Noise10/Opus24 component scores and a nuisance-regression screen. Noise5 and Opus64 cannot substitute for those components. The 9107 adjusted order-score class AUC was 0.538 for ERM and 0.432 for channel in the declared direction. This does not identify an independent forensic cue. The 9207 result repeats order asymmetry at the decision level, with no matched-component or nuisance-screen replication.

Ở B clean, 9107 có LA_0100 (0→6) và LA_0107 (5→9), chiếm 10/12 báo động tăng ròng. 9207 có giảm rải ở nhiều speaker, LA_0108 giảm 3→1 và LA_0103 tăng 3→4. Tệp số liệu giữ denominator từng speaker và condition.

## Quyết định thực nghiệm và JAES

**SEED_9307_NO_GO.** Protocol khóa yêu cầu hai seed 9107/9207, còn 9307 là tùy chọn. Kế hoạch ba cặp trước đó là dự kiến trước khi khóa protocol. Cặp mới có thể giúp đo độ nhạy initialization, nhưng không tạo cohort độc lập hoặc xác lập C2. Hai cặp đủ để chuyển sang bước chuẩn bị đánh giá đã đăng ký. Hai arm 9207 đã ghi khoảng 205 phút optimizer và validation cộng lại. Giới hạn hoàn thành tùy chọn là 16/11, và snapshot hiện bị chặn bởi commit. Ưu tiên ngân sách cho held-out và QA. Medium-plus và foundation NO-GO cho bản thảo tháng 12.

**JOURNAL_CONTRIBUTION_INSUFFICIENT** cho nộp ngay. C1 là đóng góp hỗ trợ empirical và engineering, C2_DIAGNOSTIC_REJECT, C3 deferred. Hai seed hỗ trợ câu hỏi về trade-off ở detector nhẹ dưới hai chuỗi xử lý. Phần còn thiếu nhỏ nhất là kiểm tra locked held-out, báo cả kết quả bất lợi và đối chiếu rõ với nghiên cứu đã công bố. Phần mở rộng conference cần trích dẫn venue/DOI nếu có và xác minh quyền tái sử dụng. JAES yêu cầu abstract 100–200 từ và đánh giá mức mở rộng so với conference. [AES Journal Author Guidelines](https://aes.org/publications/journal-author-guidelines/)

## Held-out

**HELDOUT_NOT_AUTHORIZED.** Prelock giữ đúng bốn model đã chọn và đề xuất 1.088 nguồn, 17.408 prediction. Fallback 544 nguồn, 8.704 prediction chỉ theo tài nguyên trước khi mở nhãn. Benchmark development ngoại suy khoảng 50/25 phút, chưa gồm kiểm tra hash, truy cập ZIP, checkpoint, IO và bootstrap. Dung lượng tại snapshot khoảng 13.44 GiB C, 19.35 GiB D, RAM khả dụng 1.22 GiB và commit khả dụng 0.29 GiB. Giải phóng bộ nhớ và xác minh lại gate 2 GiB RAM, 3 GiB commit trước đánh giá. Zip audit chỉ đọc directory metadata và hash archive. Chưa đọc nội dung official protocol, nhãn, audio hoặc license. Cần tác giả cho phép riêng việc đọc protocol/kiểm tra overlap và việc thực thi locked scoring. Sau chấp thuận truy cập, đối chiếu số trial CM với danh mục ZIP, kiểm tra license, ID, speaker, checksum và lineage nếu có. A16/A19 dùng lại thuật toán của A04/A06, A17 chia sẻ framework với A05. [ASVspoof 2019 documentation](https://arxiv.org/html/1911.01601)

Lệnh sau chỉ dành cho giai đoạn đã có lock được tác giả duyệt. Cùng lock cho phép tiếp tục từ các shard nguồn đã commit, và runner từ chối nếu metrics hoàn tất đã tồn tại.

Lần đo lại sau khi tạo prelock vẫn báo `SYSTEM_COMMIT_PRESSURE`, với 1,76 GiB RAM khả dụng và 2,15 GiB commit khả dụng. Snapshot trong prelock giữ nguyên số đo tại thời điểm tạo. Trước khi cho phép chạy, cần tạo bản lock riêng có bản ghi phê duyệt và lý do chọn subset dựa trên tài nguyên. CLI hiện có gộp kiểm tra protocol và chấm điểm sau khi duyệt, nên chưa có lệnh riêng để chạy bước đọc protocol mà không thể tiếp tục sang inference. Giai đoạn duyệt truy cập protocol trước cần được xử lý riêng.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.empirical_heldout --approved-lock outputs/jaes_v2/phase10/empirical_execution_v1/heldout_approved_9107_9207_v1.json
```

## Terminal và lịch

Monitor đọc status/contract ở terminal riêng, không khởi chạy hoặc điều khiển trainer. Trường thời gian elapsed là thời gian monitor vì trainer không lưu wall-clock xuyên resume. ETA chỉ tính optimizer. Status RUNNING cũ hiện cảnh báo kiểm tra terminal gốc, vì tệp không đủ để phân biệt validation dài với process exit. Validation hiển thị lần commit mới nhất, có thể thuộc checkpoint cuối, khác model đã chọn.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.medium_progress --experiment p10_medium_channel_s9207_v2 --once
```

Ví dụ đã kiểm tra với run thật: Epoch 21/21, Updates 840/840, Best checkpoint u000720_e018_c000, Status SCIENTIFIC_COMPLETE. Bỏ --once để theo dõi một run đang hoạt động. Dừng monitor bằng Ctrl+C.

12 test đã đạt bằng Python hệ thống, gồm test monitor, empirical execution và final score diagnostic. Môi trường huấn luyện `.venv` thiếu pytest. Prelock bị từ chối đúng tại gate authorization trước truy cập ZIP. Không thực hiện training, scoring held-out hay calibration trong đợt này.

10–13/11 chốt audit và bảng. Đến 16/11 giữ quyết định dừng medium. 17–20/11 chốt resource và xin phép hai bước. 21–30/11 thực thi locked evaluation nếu các gate đạt. 01–07/12 phân tích. 08–14/12 hoàn tất related work, references và disclosure. 15–20/12 QA và gói nộp.

[Protocol](EMPIRICAL_EXECUTION_PROTOCOL.md) · [Kết quả số](../../outputs/jaes_v2/phase10/empirical_execution_v1/post9207_consolidated_v1.json) · [Prelock](../../outputs/jaes_v2/phase10/empirical_execution_v1/heldout_prelock_9107_9207_v1.json)
