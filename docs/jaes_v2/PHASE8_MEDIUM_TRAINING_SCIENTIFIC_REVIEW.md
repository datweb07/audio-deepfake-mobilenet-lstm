# Phase 8 — đánh giá khoa học sau hai lượt medium

Ngày 08/11/2026, Asia/Novosibirsk. Cả hai baseline đã hoàn tất trên development ASVspoof 2019 LA, seed 6107, ngưỡng FAKE ≥ 0.5. Channel cải thiện chất lượng xếp hạng ở ba stress chính, đồng thời tăng clean false alarm ở B và làm nặng thêm lỗi của một số attack. Các kết quả là so sánh mô tả của một cặp model đã chọn.

## Kiểm toán và phạm vi

- ERM và channel cùng 640 nguồn train (320 REAL, 320 FAKE), A gồm 224 nguồn development (112+112, 7 speaker), B gồm 144 REAL (9 speaker). Mỗi model có 2.576 score, đủ 368 nguồn × 7 điều kiện. Speaker train và development tách nhau theo audit. A01–A06 đã xuất hiện trong train. Generator checkpoint và lineage toàn cục chưa được xác minh.
- Kiểm lại toàn bộ hash seal, hash từng seal checkpoint, liên kết journal → controller, latest pointer, epoch được chọn và file model, manifest, hash từng audio nguồn train/development, config đã đăng ký và nhãn/stratum/threshold từng score. Dựng lại 14 hàng metric và 42 hàng attack mỗi model từ score. ERM có 993 file trong seal và 43 checkpoint, channel có 1.181 file và 104 checkpoint. Calibration và official evaluation để nguyên phạm vi đóng.
- Manifest SHA-256: `029ca108fe05f14971c5dba02468fe555699ab41871f94d443184a63c44df319`. Source fingerprint chung: `9efe0d683f046b9d762e1ccc6ac44637767b55e4f830ecf44532e5d4b37f7a2c`. Hai run cùng initialization hash và thứ tự nguồn qua 840 update. Mỗi run có 13.440 view có nhãn. ERM toàn clean. Channel có 3.310 clean, 3.386 gain_half, 3.331 noise10, 3.413 opus24.
- Fingerprint các dependency dùng chung, kể cả FFmpeg, khớp bytes hiện tại. Trainer có nhiều phiên bản do phục hồi bộ nhớ. ERM ghi nhận hai trainer hash, channel ghi nhận bốn. Các hash lịch sử đối chiếu với allowlist migration trong trainer, còn hash channel cuối đối chiếu bytes hiện tại. Kiểm toán này không tái dựng binary hoặc replay từng phiên bản trainer lịch sử. Channel bắt đầu tính gradient theo segment ở step 426 và pointwise matmul ở step 430. ERM dùng gradient gộp. Hai export cuối đều được chấm bằng CNN theo segment. Vì đường tính gradient và validation trong lịch sử có khác biệt số thực, đối chiếu này chưa cô lập hoàn toàn hiệu ứng augmentation.
- Bằng chứng chính nằm trong [ERM run](../../outputs/jaes_v2/phase8/training/p8_medium_erm_partial_s6107_v1/) và [channel run](../../outputs/jaes_v2/phase8/training/p8_medium_channel_partial_s6107_v1/). Script audit chỉ đọc [phase8_medium_review.py](../../scripts/jaes_v2/phase8_medium_review.py). JSON bàn giao giữ hash, checkpoint, số liệu chi tiết và lịch sử.

## Học, chọn checkpoint và fine-tuning

Hai resolved config chỉ khác experiment ID và augmentation policy. Kiểm tra runtime, checkpoint và journal cho thấy cùng lịch nguồn, còn các đường tính số học khác nhau như mô tả ở trên.

Cả hai chạy đủ 21 epoch gồm một epoch warmup và 20 epoch partial, 40 update/epoch. ERM chọn epoch 17, step 680, clean A NLL 0.125346. Channel chọn epoch 16, step 640, NLL 0.162973. ERM kết thúc vì hết lịch với 4 epoch liên tiếp chưa cải thiện. Channel chạm patience 5 tại epoch 21, trùng giới hạn lịch, nên patience không rút ngắn ngân sách. NLL ERM giảm từ 0.33255 xuống 0.12535 tại điểm tốt nhất rồi tăng lên 0.14091 ở cuối, đạt 0.18733 tại epoch 20 dù train loss vẫn thấp. Channel có hình thái tương tự, best 0.16297 rồi 0.17113 cuối. Đây là dấu hiệu overfitting muộn trên development đã dùng để chọn model và cho thấy lợi ích giảm dần. Chuỗi dao động và giới hạn 21 epoch chưa xác nhận hội tụ toán học hoặc ổn định giữa seed. Train loss của channel bao gồm các view khác clean nên so sánh chéo mức loss cần thận trọng.

Đối chiếu trực tiếp tensor trong export epoch 1 và export được chọn xác nhận mỗi run có **6/206 tensor backbone thay đổi**, gồm bốn kernel và hai bias của bốn lớp Conv2D cuối. **0/136 tensor BatchNorm thay đổi**. Thay đổi trọng số chứng minh fine-tuning hoạt động. Nó chưa tách tác động của representation khỏi lịch học, head và dữ liệu đầu vào. Hai run cùng fine-tune nên không có đối chứng frozen/partial có cùng ngân sách ở Phase 8.

Hàng dưới là trung bình loss của 40 update train/epoch trong journal và NLL clean A trên 224 nguồn từ history.

| Epoch | ERM train | ERM dev | Channel train | Channel dev |
| --- | --- | --- | --- | --- |
| 1 | 0.55029 | 0.33255 | 0.62599 | 0.42871 |
| 2 | 0.31541 | 0.24573 | 0.47420 | 0.31936 |
| 3 | 0.24642 | 0.20564 | 0.39081 | 0.27106 |
| 4 | 0.22783 | 0.22388 | 0.36946 | 0.26470 |
| 5 | 0.18568 | 0.17702 | 0.34575 | 0.23114 |
| 6 | 0.16322 | 0.18596 | 0.31854 | 0.21526 |
| 7 | 0.17879 | 0.15716 | 0.30238 | 0.20626 |
| 8 | 0.12603 | 0.15080 | 0.28215 | 0.19763 |
| 9 | 0.10837 | 0.17440 | 0.24162 | 0.20600 |
| 10 | 0.10278 | 0.17768 | 0.25146 | 0.26739 |
| 11 | 0.09043 | 0.14097 | 0.26544 | 0.18131 |
| 12 | 0.07437 | 0.14608 | 0.20799 | 0.17639 |
| 13 | 0.07061 | 0.16312 | 0.22338 | 0.17685 |
| 14 | 0.05326 | 0.17997 | 0.21001 | 0.17988 |
| 15 | 0.06268 | 0.12700 | 0.20856 | 0.18103 |
| 16 | 0.03923 | 0.12797 | 0.19386 | 0.16297 |
| 17 | 0.03631 | 0.12535 | 0.20023 | 0.17641 |
| 18 | 0.02632 | 0.12558 | 0.19101 | 0.19266 |
| 19 | 0.03308 | 0.15320 | 0.19523 | 0.18510 |
| 20 | 0.04824 | 0.18733 | 0.16593 | 0.16796 |
| 21 | 0.02722 | 0.14091 | 0.17263 | 0.17113 |

## Metrics theo điều kiện

FA và miss ghi dạng số lỗi/số lượt có nhãn tương ứng. Accuracy và F1 lớp FAKE dùng 224 nguồn A mỗi điều kiện, trong đó F1 = 2TP/(2TP+FP+FN). ROC-AUC xếp hạng 112 REAL và 112 FAKE, tương ứng 12.544 cặp có thể so sánh mỗi điều kiện. PR-AUC là average precision của sklearn. EER dùng điểm gần nhất theo |FPR−FNR| trong routine Phase 7. Ba metric xếp hạng dùng raw scores. Tất cả quyết định nhị phân ở 0.5. Với B chỉ báo cáo FA (và accuracy có thể suy ra bằng 1−FA). Attack/B một lớp để trống AUC, AP, EER và F1 hai lớp.

Stress chính theo Phase 7 gồm **noise5, A_then_B, B_then_A**. Hàng stress lấy trung bình bằng nhau của ba metric từng điều kiện, không gộp score để tính lại AUC/EER/F1. Số lỗi stress là tổng trên 336 view mỗi lớp A và 432 view REAL B. View lặp cùng nguồn nên các tổng này không phải số quan sát độc lập. Opus64 và hai gain là điều kiện phụ, giữ từng hàng riêng. Nguồn: hai `scores.csv`, đối chiếu `metrics.json`.

| Điều kiện | Run | A REAL FA | A FAKE miss | A acc | A F1 | A ROC-AUC | A AP | A EER | B REAL FA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | erm | 4/112 (3.57%) | 8/112 (7.14%) | 94.64% | 94.55% | 98.87% | 99.13% | 5.36% | 19/144 (13.19%) |
| clean | channel | 4/112 (3.57%) | 8/112 (7.14%) | 94.64% | 94.55% | 97.91% | 98.44% | 5.36% | 30/144 (20.83%) |
| gain_half | erm | 0/112 (0.00%) | 16/112 (14.29%) | 92.86% | 92.31% | 98.21% | 98.63% | 4.46% | 2/144 (1.39%) |
| gain_half | channel | 0/112 (0.00%) | 15/112 (13.39%) | 93.30% | 92.82% | 97.92% | 98.51% | 6.25% | 7/144 (4.86%) |
| gain_quarter | erm | 3/112 (2.68%) | 21/112 (18.75%) | 89.29% | 88.35% | 92.99% | 95.04% | 16.96% | 2/144 (1.39%) |
| gain_quarter | channel | 2/112 (1.79%) | 18/112 (16.07%) | 91.07% | 90.38% | 93.98% | 95.71% | 15.18% | 5/144 (3.47%) |
| noise5 | erm | 15/112 (13.39%) | 87/112 (77.68%) | 54.46% | 32.89% | 57.10% | 62.44% | 41.96% | 10/144 (6.94%) |
| noise5 | channel | 17/112 (15.18%) | 52/112 (46.43%) | 69.20% | 63.49% | 77.98% | 82.60% | 28.12% | 22/144 (15.28%) |
| opus64 | erm | 6/112 (5.36%) | 3/112 (2.68%) | 95.98% | 96.04% | 98.81% | 99.11% | 3.57% | 36/144 (25.00%) |
| opus64 | channel | 6/112 (5.36%) | 7/112 (6.25%) | 94.20% | 94.17% | 98.09% | 98.57% | 6.25% | 38/144 (26.39%) |
| A_then_B | erm | 94/112 (83.93%) | 13/112 (11.61%) | 52.23% | 64.92% | 67.72% | 74.01% | 37.05% | 98/144 (68.06%) |
| A_then_B | channel | 38/112 (33.93%) | 29/112 (25.89%) | 70.09% | 71.24% | 82.45% | 86.44% | 26.34% | 47/144 (32.64%) |
| B_then_A | erm | 28/112 (25.00%) | 62/112 (55.36%) | 59.82% | 52.63% | 61.51% | 67.67% | 46.43% | 14/144 (9.72%) |
| B_then_A | channel | 17/112 (15.18%) | 47/112 (41.96%) | 71.43% | 67.01% | 82.09% | 85.81% | 25.89% | 24/144 (16.67%) |
| stress_macro | erm | 137/336 (40.77%) | 162/336 (48.21%) | 55.51% | 50.15% | 62.11% | 68.04% | 41.82% | 122/432 (28.24%) |
| stress_macro | channel | 72/336 (21.43%) | 128/336 (38.10%) | 70.24% | 67.25% | 80.84% | 84.95% | 26.79% | 93/432 (21.53%) |

Channel giảm A stress FA **137/336 → 72/336** (−19.35 pp), miss **162/336 → 128/336** (−10.12 pp), và B stress FA **122/432 → 93/432** (−6.71 pp). Stress macro A ROC-AUC **0.6211 → 0.8084** và EER **41.82% → 26.79%**. Cả ba AUC stress đều tăng và EER giảm, vì vậy lợi ích xếp hạng vượt phần giải thích bằng một dịch chuyển score đơn điệu chung. Operating point vẫn đóng vai trò lớn. Mean score A tăng 0.3315→0.5034 ở noise5 nhưng giảm 0.7711→0.5825 ở A_then_B. A_then_B giảm FA 94→38 song tăng miss 13→29. Noise5 giảm miss 87→52 song tăng FA 15→17 ở A và 10→22 ở B. B_then_A giảm hai lỗi A, còn FA ở B tăng 14→24. Opus64 A miss 3→7 và EER 3.57%→6.25%. Không có cơ sở gộp tất cả thành cải thiện đồng đều dưới codec/noise.

Clean A giữ nguyên 4/112 FA và 8/112 miss, EER 5.36%, nhưng ROC-AUC giảm **0.98868→0.97911**, AP giảm 0.99127→0.98444 và NLL tăng. Clean A đạt caps thăm dò **FA ≤10%, miss ≤15%** ở cả hai run. Cap này chỉ được áp dụng hậu kiểm vì checkpoint đã đăng ký CLEAN_BCE. B clean FA **19/144→30/144** (+7.64 pp). Nếu đặt cùng cap FA 10% cho B, cả hai vượt cap. B không có cap chọn checkpoint đã đăng ký và mức tăng này là chi phí ngoài A cần được khóa thành tiêu chí trong protocol tiếp theo. Với biên delta +2 pp của Phase 7 dùng làm tham chiếu hậu kiểm, A clean delta bằng 0 còn B FA tăng vượt biên. Không điều chỉnh ngưỡng.

## Attack, speaker và flip

Bảng attack dùng FAKE miss tại 0.5, theo `attack_metrics.json` và `scores.csv`. Mỗi ô là ERM → channel. Stress gộp lỗi trên ba view mỗi nguồn. Worst attack stress của cả hai là **A05** (40/54→47/54). Worst clean là **A06** (4/18→6/18). A02 stress không đổi 35/57 nhưng chuyển lỗi giữa điều kiện, còn A06 stress xấu hơn 19/54→37/54. A01, A03 và A04 hưởng lợi lớn trên cohort này. Cỡ nhóm 18–19 nguồn không cho phép khẳng định hiệu ứng bền qua generator.

| Attack | Clean | Gain half | Gain quarter | Noise5 | Opus64 | A→B | B→A | Stress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A01 | 0/19 (0.00%) → 0/19 (0.00%) | 0/19 (0.00%) → 0/19 (0.00%) | 0/19 (0.00%) → 0/19 (0.00%) | 13/19 (68.42%) → 1/19 (5.26%) | 0/19 (0.00%) → 0/19 (0.00%) | 0/19 (0.00%) → 0/19 (0.00%) | 4/19 (21.05%) → 0/19 (0.00%) | 17/57 (29.82%) → 1/57 (1.75%) |
| A02 | 1/19 (5.26%) → 0/19 (0.00%) | 0/19 (0.00%) → 0/19 (0.00%) | 0/19 (0.00%) → 0/19 (0.00%) | 19/19 (100.00%) → 14/19 (73.68%) | 0/19 (0.00%) → 0/19 (0.00%) | 0/19 (0.00%) → 6/19 (31.58%) | 16/19 (84.21%) → 15/19 (78.95%) | 35/57 (61.40%) → 35/57 (61.40%) |
| A03 | 1/19 (5.26%) → 0/19 (0.00%) | 1/19 (5.26%) → 0/19 (0.00%) | 1/19 (5.26%) → 0/19 (0.00%) | 16/19 (84.21%) → 6/19 (31.58%) | 0/19 (0.00%) → 0/19 (0.00%) | 6/19 (31.58%) → 0/19 (0.00%) | 12/19 (63.16%) → 1/19 (5.26%) | 34/57 (59.65%) → 7/57 (12.28%) |
| A04 | 2/19 (10.53%) → 2/19 (10.53%) | 2/19 (10.53%) → 2/19 (10.53%) | 3/19 (15.79%) → 1/19 (5.26%) | 11/19 (57.89%) → 1/19 (5.26%) | 0/19 (0.00%) → 1/19 (5.26%) | 0/19 (0.00%) → 0/19 (0.00%) | 6/19 (31.58%) → 0/19 (0.00%) | 17/57 (29.82%) → 1/57 (1.75%) |
| A05 | 0/18 (0.00%) → 0/18 (0.00%) | 0/18 (0.00%) → 0/18 (0.00%) | 0/18 (0.00%) → 0/18 (0.00%) | 17/18 (94.44%) → 17/18 (94.44%) | 0/18 (0.00%) → 0/18 (0.00%) | 7/18 (38.89%) → 14/18 (77.78%) | 16/18 (88.89%) → 16/18 (88.89%) | 40/54 (74.07%) → 47/54 (87.04%) |
| A06 | 4/18 (22.22%) → 6/18 (33.33%) | 13/18 (72.22%) → 13/18 (72.22%) | 17/18 (94.44%) → 17/18 (94.44%) | 11/18 (61.11%) → 13/18 (72.22%) | 3/18 (16.67%) → 6/18 (33.33%) | 0/18 (0.00%) → 9/18 (50.00%) | 8/18 (44.44%) → 15/18 (83.33%) | 19/54 (35.19%) → 37/54 (68.52%) |

Worst speaker là tỷ lệ lỗi lớn nhất trong từng stratum/lớp, giữ đồng hạng. Các mẫu số 16 clean và 48 stress tương ứng 16 nguồn mỗi speaker/lớp và ba view stress. B không có FAKE. Nguồn: ghép `speaker_id`, `label` và `condition_id` trong score.

| Nhóm (0 REAL, 1 FAKE) | ERM | Channel |
| --- | --- | --- |
| A_PAIRED/0/clean | LA_0077 3/16 (18.75%) | LA_0077 4/16 (25.00%) |
| A_PAIRED/0/stress | LA_0071 27/48 (56.25%) | LA_0070 21/48 (43.75%) |
| A_PAIRED/1/clean | LA_0069 2/16 (12.50%), LA_0077 2/16 (12.50%) | LA_0069 2/16 (12.50%), LA_0071 2/16 (12.50%), LA_0072 2/16 (12.50%) |
| A_PAIRED/1/stress | LA_0069 28/48 (58.33%), LA_0077 28/48 (58.33%) | LA_0075 21/48 (43.75%) |
| B_REAL_ONLY/0/clean | LA_0102 4/16 (25.00%), LA_0107 4/16 (25.00%) | LA_0107 11/16 (68.75%) |
| B_REAL_ONLY/0/stress | LA_0103 24/48 (50.00%) | LA_0102 26/48 (54.17%) |

Flips ghép theo cùng sample ID, tính đúng→sai và sai→đúng so với clean của chính model. Mẫu số là số nguồn của lớp trong stratum ở mỗi điều kiện. Hai chiều có mẫu số chung và có thể kiểm lại bằng stress errors = clean errors + đúng→sai − sai→đúng.

| Điều kiện/stratum/lớp | ERM đúng→sai | ERM sai→đúng | Channel đúng→sai | Channel sai→đúng |
| --- | --- | --- | --- | --- |
| gain_half/A_PAIRED/0 | 0/112 | 4/112 | 0/112 | 4/112 |
| gain_half/A_PAIRED/1 | 10/112 | 2/112 | 7/112 | 0/112 |
| gain_half/B_REAL_ONLY/0 | 0/144 | 17/144 | 0/144 | 23/144 |
| gain_quarter/A_PAIRED/0 | 3/112 | 4/112 | 2/112 | 4/112 |
| gain_quarter/A_PAIRED/1 | 15/112 | 2/112 | 11/112 | 1/112 |
| gain_quarter/B_REAL_ONLY/0 | 1/144 | 18/144 | 3/144 | 28/144 |
| noise5/A_PAIRED/0 | 15/112 | 4/112 | 16/112 | 3/112 |
| noise5/A_PAIRED/1 | 80/112 | 1/112 | 46/112 | 2/112 |
| noise5/B_REAL_ONLY/0 | 8/144 | 17/144 | 12/144 | 20/144 |
| opus64/A_PAIRED/0 | 2/112 | 0/112 | 2/112 | 0/112 |
| opus64/A_PAIRED/1 | 0/112 | 5/112 | 0/112 | 1/112 |
| opus64/B_REAL_ONLY/0 | 17/144 | 0/144 | 9/144 | 1/144 |
| A_then_B/A_PAIRED/0 | 92/112 | 2/112 | 36/112 | 2/112 |
| A_then_B/A_PAIRED/1 | 12/112 | 7/112 | 24/112 | 3/112 |
| A_then_B/B_REAL_ONLY/0 | 82/144 | 3/144 | 32/144 | 15/144 |
| B_then_A/A_PAIRED/0 | 28/112 | 4/112 | 16/112 | 3/112 |
| B_then_A/A_PAIRED/1 | 57/112 | 3/112 | 42/112 | 3/112 |
| B_then_A/B_REAL_ONLY/0 | 10/144 | 15/144 | 13/144 | 19/144 |

## Đọc kết quả cùng Phase 7 và chi phí

Phase 7 F0/F1 là đối chứng matched head-only/partial 40 update. AUC clean tăng 0.7508→0.8104, nhưng stressed FAKE miss tăng 0.6756→0.7173. Phase 7 H3–H2 có delta miss stress trung bình −0.00397 và SD seed 0.08174, cả ba block trượt guardrail A clean đã đăng ký. Các baseline H0/H1 dùng backbone frozen với seed/budget riêng. Ngay cả khi dùng nguồn thuộc cùng selector, ngân sách, trainable policy, lựa chọn model và số seed khác nhau. Chênh lệch liên phase chỉ có giá trị bối cảnh. Phase 8 cho thấy head và backbone học được trên lịch dài hơn, nhưng chưa cho thấy lợi ích representation độc lập hoặc generalization sang generator/corpus mới. Hướng cần kiểm tra là liệu cải thiện xếp hạng còn giữ được dưới matched multi-seed và khi kiểm soát nuisance của thứ tự xử lý. [Phase 7 fine-tuning](PHASE7_FINE_TUNING_RESULTS.md) · [Phase 7 multi-seed](PHASE7_MULTI_SEED_RESULTS.md)

`resource_measurements.json` đo **invocation cuối**, không đo toàn bộ lịch train. ERM 877.65 s, 0 update, peak 548.33 MiB, checkpoint bytes 450.94 MB. Channel 610.17 s, 3 update (24.81 s đường update), peak 622.47 MiB, checkpoint bytes 1,090.87 MB. Model forward tổng 415.90 s và 305.40 s qua lượt validation epoch cuối và final dev. Với 224 + 2.576 = 2.800 lượt forward, tương ứng khoảng 149 và 109 ms/view trong các invocation này, gồm thời gian probe nếu có. Đây chưa phải benchmark inference kiểm soát môi trường, latency end-to-end hay tổng chi phí training. Cả hai gặp giới hạn bộ nhớ, channel thêm chi phí tái khởi động worker, tái tính gradient và I/O codec. Không suy luận channel nhanh hơn từ hai invocation có lịch sử và điều kiện máy khác nhau.

## Novelty và quyết định

Khoảng trống thực nghiệm trong LAVA là giữ tỷ lệ báo nhầm giọng thật trên nhóm B và giảm lỗi A05/A06 khi hai phép xử lý đổi thứ tự, trong khi AUC stress cải thiện. Evidence hiện có chưa phân biệt mất cue do channel, shortcut codec/noise, khác biệt speaker, hay tác động của cohort. Các phương pháp đã biết xử lý những phần liên quan của vấn đề. Chưa có so sánh matched chứng minh chúng không giải quyết được trường hợp này. Đối chiếu tài liệu gốc, giả thuyết có thể bác bỏ và yêu cầu dữ liệu ở [hướng Phase 9](PHASE9_NOVELTY_RESEARCH_DIRECTION.md).

| Gate | Kết quả | Căn cứ |
| --- | --- | --- |
| MEDIUM_ERM_VALIDATED | YES | Sealed 840 updates, 2,576 paired scores, verified checkpoint selection and changed backbone tensors. Validation applies to this development baseline. |
| MEDIUM_CHANNEL_VALIDATED | YES | Sealed 840 updates and complete score coverage. The baseline is usable on development. Its B clean false-alarm increase and different numerical gradient path constrain the conclusion. |
| MULTI_SEED_EXPANSION_JUSTIFIED | YES | A +18.73 percentage point stress macro ROC-AUC signal with a B clean cost warrants narrowly paired replication at 6207 and 6307 after locking identical numerical, data, and selection policy. This is justification to test a trade-off, not a success declaration or execution authorization. |
| FULL_FOUNDATION_TRAINING_JUSTIFIED | NO | Single seed, reused development, B clean false alarms 30/144 and A05 stressed misses 47/54 for channel. Larger scale does not resolve the controlled-comparison or transport question. |
| NOVEL_C2_METHOD_IDENTIFIED | NO | Novelty not established. Intervention-targeted evidence survival is a falsifiable hypothesis pending identified paired cues and comparison to shortcut intervention and consistency literature. |
| NOVEL_C2_METHOD_READY_FOR_TRAINING | NO | Requires source-aligned cue data, a validated survival target, matched controls and an independent protocol. |

**NOVELTY_NOT_ESTABLISHED.** Hai YES đầu chỉ xác nhận baseline development đã thực thi và đo được. YES multi-seed là quyết định khoa học cho một replication hẹp sau khi khóa protocol, không phải lệnh chạy hoặc kết luận hiệu quả. Không chạy thêm training trong vòng review.

## Tệp bàn giao

| Run | Metrics | Attack metrics | History |
| --- | --- | --- | --- |
| ERM | [metrics.json](../../outputs/jaes_v2/phase8/training/p8_medium_erm_partial_s6107_v1/metrics.json) | [attack_metrics.json](../../outputs/jaes_v2/phase8/training/p8_medium_erm_partial_s6107_v1/attack_metrics.json) | [history.json](../../outputs/jaes_v2/phase8/training/p8_medium_erm_partial_s6107_v1/history.json) |
| Channel | [metrics.json](../../outputs/jaes_v2/phase8/training/p8_medium_channel_partial_s6107_v1/metrics.json) | [attack_metrics.json](../../outputs/jaes_v2/phase8/training/p8_medium_channel_partial_s6107_v1/attack_metrics.json) | [history.json](../../outputs/jaes_v2/phase8/training/p8_medium_channel_partial_s6107_v1/history.json) |

[Decision JSON](PHASE8_POST_TRAINING_DECISION.json) · [Master report sau training](LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT_POST_TRAINING.md)
