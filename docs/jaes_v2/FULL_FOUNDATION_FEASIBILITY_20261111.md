# Khả thi kỹ thuật của ASVspoof full foundation

**11/11/2026 — NO-GO cho manuscript tháng 12.** Manifest và audio train đã được xác minh. Vòng đời full foundation, khả năng lưu checkpoint và bằng chứng chất lượng khi tăng quy mô chưa đạt điều kiện.

## Dataset và workload

| Vai trò | Số bản ghi | REAL | FAKE | Speaker |
| --- | ---: | ---: | ---: | ---: |
| Foundation train | 25.380 | 2.580 | 22.800 | 20 |
| Foundation development | 17.503 | 2.023 | 15.480 | 16 |
| Calibration dự trữ | 7.341 | 525 | 6.816 | 4 |
| Medium train | 640 | 320 | 320 | Theo protocol medium |

SHA-256 của train manifest là `7d24153c35c3867b02a5f99ee60b8ba6c0a67bfbb910e69f6e68611f28796e72`. Cả ba manifest khớp bản đăng ký, không giao nhau theo ID, speaker chính thức và SHA-256. Tất cả 25.380 file train có hash byte khớp manifest. Đây là xác minh local. Source utterance, generator lineage và decoded/perceptual duplicates vẫn có giới hạn. Calibration tiếp tục ở vai trò dự trữ. [Manifest và metadata](../../outputs/jaes_v2/datasets/asv2019_foundation_v1)

Medium có 50% FAKE, foundation có 89,83%. Natural sampling xáo trộn nguồn theo epoch với seed xác định. Nếu giữ batch 16 và 21 epoch, số update giả định là `ceil(25380/16) × 21 = 33327` mỗi arm, gồm batch cuối 4 nguồn. Một cặp là 66.654 update và hai seed là 133.308. Đây là kịch bản tính chi phí. Chính sách tỷ lệ lớp, quota attack, loss weighting, partial batch và LR cần đăng ký riêng. Trong medium cả ERM và channel dùng cùng BCE, channel lấy một view từ clean, gain-half, Noise10 hoặc Opus24, với seed/cursor xác định. Class weights trong config là tham số cho các objective khác. Phase 10 đang chạy microbatch 1, gradient accumulation theo batch, Adam, warm-up một epoch rồi partial fine-tune 20 epoch với LR giảm tuyến tính theo epoch. LRU tensor tối đa 8 mục, không precompute toàn bộ foundation.

## Engine và điều kiện nghiệm thu

- CLI medium khóa manifest, registry, 840 update và scope khoa học medium. CLI Phase 10 riêng chỉ có profile qualification `lifecycle` và `100`. Config Phase 8 và engine dùng lại các thành phần có thể hỗ trợ việc thiết kế một profile tương lai. Hiện chưa có CLI full foundation được nghiệm thu.
- Medium dùng CLEAN_BCE, fixed budget, checkpoint mỗi 10 update và cuối epoch, Tier A clean mỗi epoch, Tier B mỗi ba epoch và cuối run, Tier C trên selected export. Bad-epoch counter được ghi nhưng đường fixed-budget không dùng nó để dừng sớm. Resume khôi phục model, optimizer, cursor, lịch sử và counter, rồi kiểm tra hash. Dữ liệu full cần quyết định riêng về dev size, validation cadence, checkpoint retention và invocation dài. Giữ Tier B/C trên cohort đã khóa giúp duy trì so sánh. Dùng toàn bộ 17.503 dev là một workload khác.
- `FULL_SHAPE_RECOMPUTE` giữ shape CNN gốc trong gradient. Bằng chứng lịch sử v2 đạt gradient parity trên 9 fixture thật, exact resume 9 update có restart tại update 5, và 100-update comparator có restart tại update 55. Model/optimizer/journal/score khớp ở mức logic. Khác biệt metadata serialization không được dùng làm kết luận khác weights. Full-development parity có 2.576 score pairs và 0 decision flips. Bốn selected export medium vẫn khớp seal. Engine kiểm tra nonfinite loss, gradient, weights, gradient zero/exploding, và dừng khi resource gate thất bại. Những phép thử này xác nhận medium và phạm vi fixture đã kiểm. Full-foundation lifecycle, long-run resource behavior và export sau một run lớn chưa có bằng chứng.
- F1 được kiểm bằng config/manifest, fixture transform, model restore trên tensor tổng hợp, objective/optimizer fixture, status và resource-stop tests. F2 dùng bằng chứng bounded lifecycle đã có. Một F2 tương lai chỉ nên chạy trong profile cách ly với ID mới và phép thử vài update, có batch ngắn, resume sau một checkpoint, validation và export. So sánh với uninterrupted control, kiểm tra hash trạng thái logic, full-shape parity, RAM/commit và footprint. Nếu chưa có profile full cách ly, dừng ở thiết kế và fixture.
- F3 yêu cầu independent train-derived development folds, một giả thuyết tăng quy mô định trước, matched ERM và Channel cùng tỷ lệ lớp/budget, rồi tập xác nhận riêng cho mô hình mới. Held-out đã mở có thể cung cấp chẩn đoán tiếp theo với nhãn exploratory. Smoke test và một pilot thuận lợi chưa dự báo được chất lượng sau 21 epoch.

## Tài nguyên và thời gian

Snapshot preflight đạt trên Windows CPU, Python 3.11.9, TF/Keras 2.15. RAM vật lý khoảng 15,3 GiB. Snapshot đầu lượt có 3,28 GiB RAM khả dụng, 13,05 GiB commit, C 9,70 GiB và D 19,53 GiB trống. Snapshot trong JSON audit có 5,42 GiB RAM khả dụng, 10,30 GiB commit, C 9,38 GiB và D 19,53 GiB trống. Gate hiện hành yêu cầu tối thiểu 2 GiB RAM, 3 GiB commit, C 3 GiB, D 15 GiB cộng 128 MiB. RTX 3050 Laptop 4 GiB là cấu hình đã ghi trước đây. Đường chạy khoa học được đo trên CPU. Snapshot không phải bằng chứng GPU acceleration.

**Đo được.** Hai run ERM medium ghi 2.488–2.556 s cho vòng update, 2.411–2.479 s validation, 18–20 s checkpoint write. Hai run Channel ghi 4.943–4.974 s update, 2.349–2.373 s validation, 18–19 s checkpoint write. Đây là elapsed update end-to-end, bao gồm chuẩn bị view và backward. Trung bình ERM 3,00 s/update, Channel 5,90 s/update, tương ứng 0,33 và 0,17 update/s. ERM 9107 có ba invocation nên tổng có overhead khôi phục và phần work ghi lại. Archive checkpoints medium đang chiếm khoảng 0,88 GiB mỗi arm. Pilot 100 update ghi 698,51 s update, 891,88 s validation, 2,76 s checkpoint, peak RSS 724,37 MiB, 116.185.874 byte checkpoint, commit headroom tối thiểu khoảng 5,14 GiB. Các số này là đo lịch sử cho workload đã nêu. Chi phí backward độc lập từng phase và peak memory full chưa được đo.

| Kịch bản có điều kiện | Một cặp ERM + Channel | Hai cặp seed | Cách tính và giới hạn |
| --- | ---: | ---: | --- |
| Lạc quan | khoảng 81,9 giờ | 163,8 giờ | 33.327 update mỗi arm theo tốc độ medium nhanh nhất tương ứng. Chỉ tính update, giả định không giảm throughput. Đây là cận theo kịch bản, không phải cận vật lý. |
| Mốc lập kế hoạch dựa trên đo | khoảng 84,2 giờ | 168,4 giờ | Khoảng 82,4 giờ update theo mean hai seed, cộng khoảng 1,34 giờ validation nếu giữ cohort và lịch medium, khoảng 0,42 giờ checkpoint write nếu số lần tăng tương ứng. Startup và export nhỏ trong run đã đo. |
| Bảo thủ | khoảng 127 giờ | 254 giờ | Giả định update chậm hơn 50% do I/O và journal lớn hơn, cộng khoảng ba giờ cho validation, checkpoint và restart trên mỗi cặp. Full-dev scoring, phân tích và phản biện cần tính riêng. |

**Ổ đĩa là blocker trực tiếp với retention hiện tại.** Nếu giữ mỗi checkpoint khoảng 11 MB và cadence 10 update, khoảng 3.333 lần ghi mỗi arm đã tương ứng khoảng **35 GiB mỗi arm, 70 GiB mỗi cặp**, trước checkpoint cuối epoch ngoài cadence, journal tăng dần, output và reserve. D chỉ có khoảng 4,4 GiB khả dụng phía trên reserve. Tốc độ ghi có thể phi tuyến vì controller mang lịch sử. Trước một dự án full tương lai, cần policy giữ checkpoint được phê duyệt riêng, đủ storage, benchmark I/O train-only và requalification của resume. Chưa có cơ sở cam kết máy này hoàn thành full theo protocol hiện hành.

**Deadline.** 17/11 là cutoff quyết định. Một job có thể chạy nhiều ngày nhưng mốc 20/12 còn bao gồm thiết kế, nghiệm thu, chạy đối chứng, xác nhận độc lập, phân tích và supervisor review. Khối công việc đó chưa được chứng minh khả thi. Manuscript hiện tại trả lời câu hỏi đã đăng ký ở quy mô medium. Quyết định chính thức là `FULL_TRAINING_NOT_SCIENTIFICALLY_NECESSARY`.

[Bằng chứng nghiệm thu Phase 10](LAVA_JOURNAL_V2_MASTER_RESEARCH_REPORT_PHASE10.md) · [Probe và resource snapshot](../../outputs/jaes_v2/post_heldout_training_audit_v1.json)
