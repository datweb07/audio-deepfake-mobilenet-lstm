# Phase 9 — hướng nghiên cứu sau medium

08/11/2026, Asia/Novosibirsk. **NOVELTY_NOT_ESTABLISHED**. Ưu tiên phân biệt bằng chứng giả mạo còn tồn tại sau channel với dấu vết do channel tạo ra. Trên development, A05 stress miss tăng từ 40/54 lên 47/54, A06 tăng 19/54 lên 37/54 và B clean FA tăng 19/144 lên 30/144 dù stress macro AUC cải thiện. Phase 7 đã cho thấy consistency thiếu ổn định giữa seed, fine-tuning ngắn có thể cải thiện clean ranking đồng thời làm xấu stress miss. [Review Phase 8](PHASE8_MEDIUM_TRAINING_SCIENTIFIC_REVIEW.md) · [Kết quả Phase 7](PHASE7_MULTI_SEED_RESULTS.md)

## Đối chiếu prior art

Các nguồn dưới đây dùng để xác định ranh giới câu hỏi. Chưa có run LAVA matched với các bài này, vì vậy hiệu quả của chúng với đúng A05/A06 và nhóm B còn mở.

| Nguồn gốc | Phần trùng và giới hạn so sánh |
| --- | --- |
| [F-SAT, ICLR 2025](https://openreview.net/pdf?id=2GcR9bO620) | Đã có adversarial training chọn dải phổ và phân tích cue cao tần dễ tổn thương. Thêm nhiễu theo dải hoặc giữ phổ không đủ tạo khác biệt. |
| [PCL-NET, preprint tháng 10/2026](https://arxiv.org/html/2610.07216v1) | Paired consistency trên view qua neural codec. Opus truyền thống của LAVA là điều kiện khác. Giảm feature discrepancy giữa view đã có đối chứng trực tiếp. |
| [DK-CAST](https://link.springer.com/article/10.1007/s10791-025-09746-4) | Distillation từ clean teacher, codec-aware weighting, feature và phoneme alignment. Clean teacher và codec confidence đã có tiền lệ. |
| [SONAR, ICML 2026](https://proceedings.mlr.press/v306/hidekel26a.html) | Residual cao tần kết hợp representation nội dung, alignment theo lớp. Kiến trúc hai nhánh hoặc alignment bất đối xứng theo lớp đã có đối chứng mạnh. |
| [RawBoost](https://arxiv.org/abs/2111.04433) · [Group-DRO](https://arxiv.org/abs/1911.08731) | Augmentation tín hiệu và tối ưu worst-group là hai họ baseline. Thay nhóm bằng attack/condition hoặc điều chỉnh trọng số vẫn thuộc các họ này. |
| [TFCL](https://arxiv.org/html/2607.17761v2) | Đã mô hình pipeline AFE nhiều tầng và dùng soft temporal alignment cùng consistency miền tần số. Hướng order/composition cộng consistency có overlap rõ. |
| [Intervention-Based Shortcut Diagnosis, preprint 07/2026](https://arxiv.org/abs/2607.03150) | Đã dùng intervention để chẩn đoán shortcut và đánh giá lựa chọn augmentation. Chỉ làm intervention hoặc phân tích shortcut không đủ tạo khoảng cách phương pháp. |
| [GraphSeal, preprint 04/11/2026](https://arxiv.org/abs/2611.06171) · [Neural Audio Codec for Robust ADD](https://arxiv.org/abs/2609.39651) | Các hướng gần đây khai thác lan truyền dấu vết phổ-thời gian và representation codec. Đối chiếu trong vòng này ở mức nguồn tác giả, không có tái lập kết quả trên LAVA. |
| [RADAR 2026](https://arxiv.org/abs/2605.09568) · [AT-ADD](https://arxiv.org/abs/2608.23437) | Benchmark media transformation và chuyển miền/generator đã tồn tại. Một grid lớn hơn hoặc một corpus bên ngoài tự nó chưa đủ làm C2 mới. |
| [Adaptive Compression / Mel-BAP](https://arxiv.org/abs/2610.16049) | Nén representation thích ứng và perturbation chọn lọc đã được đề xuất. Adaptive weighting, high-frequency targeting và đánh đổi robustness/clean cần đối chứng tương ứng. |

## Một giả thuyết C2 để bác bỏ trước khi train

**Mục tiêu:** đo và dự đoán mức bằng chứng giả mạo *cụ thể* sống sót qua mỗi thứ tự xử lý, để phân biệt sự suy yếu của cue với score drift do channel. Đây là một giả thuyết nghiên cứu chưa vượt novelty gate.

Gọi (x_i) là tín hiệu gốc được căn chỉnh, (I_z) là can thiệp có thể bật/tắt cue đã biết, (T_o) là chuỗi xử lý với order (o) và cùng nuisance seeds, (g) là phép đo cue được khóa độc lập với detector. Với cặp (x_i^+=I_1(x_i)), (x_i^-=I_0(x_i)), định nghĩa

```text
d(i,o) = g(T_o(x_i^+)) - g(T_o(x_i^-))
r(i,o) = d(i,o) / d(i,identity), chỉ khi |d(i,identity)| > epsilon đã đăng ký
interaction(i) = d(i,A_then_B) - d(i,B_then_A)
L = mean BCE(y, sigmoid(sθ(T_o(x)))) +
    λ mean Huber([sθ(T_o(x_i^+)) - sθ(T_o(x_i^-))] - β d(i,o))
```

`sθ` là FAKE logit, `β` chuyển đơn vị phép đo sang logit và được cố định hoặc ước lượng trên train độc lập trước khi mở development mới. `r` là diagnostic, loss dùng `d` để tránh chia cho cue gốc gần 0. Cặp positive/negative và identity chỉ dùng khi thiết kế can thiệp hợp lệ. Khác biệt kỹ thuật dự kiến cần chứng minh là **target phản thực nghiệm có dấu được đo từ can thiệp cue**, có thể yêu cầu detector giảm chênh logit nếu cue bị xóa. F-SAT tạo perturbation tối ưu theo model, PCL/TFCL ép ổn định representation, DK-CAST truyền mục tiêu từ teacher và shortcut-diagnosis dùng can thiệp để phân tích. Thành phần loss trên vẫn là paired regression quen thuộc. Tính mới chỉ có thể nằm ở việc định danh và xác thực target cue hoặc estimand, với đủ bằng chứng rằng nó đem lại thông tin vượt severity và các biến thể pair đã biết.

- **Giả định:** cùng source, transcript và speaker trong cặp, cue là can thiệp duy nhất có chủ đích, nguồn và output aligned, các nuisance seeds và quy tắc chuẩn hóa được khóa. Cần generator phiên bản/checkpoint có giấy phép và bật/tắt được cue hoặc cue injection có cơ chế vật lý rõ. ASV19 A01–A06 không cung cấp pairing lineage đó. Cue tổng hợp chỉ kiểm tra nhận dạng cơ chế và không tự chuyển thành bằng chứng trên fake thật.
- **Falsification trước training:** identity/lossless phải bảo toàn target, cue knockout phải làm giảm target, gain/delay/noise seed hoặc swap order vô nghĩa không được tự giải thích nó. Lặp theo held-out speaker, source và generator. Nếu target suy ra tốt tương đương từ RMS, SNR, bitrate hoặc severity, hoặc không xác định ổn định sau căn chỉnh, dừng giả thuyết. Chốt tiêu chí đo, sai số và baseline trước khi mở cohort mới.
- **Đối chứng matched:** ERM, channel, equal-view BCE, probability/feature consistency, finite RawBoost và Group-DRO với uniform-q cùng sampling. Khóa initialization, arithmetic/validation mode, lịch update, số view và checkpoint policy. Với mục tiêu trọng tâm, đối chiếu thêm F-SAT hoặc TFCL-style được ghi rõ phần nào tái tạo được trên MobileNet.
- **Ablation bắt buộc:** λ=0, thay target bằng severity, xáo target giữ group và view count, thay cue intervention bằng nuisance-only, loại thông tin order, và không dùng cue-knockout. Báo cáo matched ranking và ngưỡng 0.5, A/B clean guardrails, A05/A06 hoặc nhóm tương ứng trên cohort mới, cùng order interaction ghép nguồn. Nếu không thắng các control có cùng ngân sách hoặc làm xấu B clean, bác bỏ cách huấn luyện.
- **MobileNetV3–LSTM:** head hiện có cung cấp logit. Cặp được tạo offline và dùng khi train. Chi phí xấp xỉ hai forward/backward có nhãn trên mỗi cặp so với một view ERM, ngoài tiền xử lý và đo cue. Đối chứng equal-view phải có cùng chi phí. Khi chỉ dùng head hiện có, inference vẫn một forward mỗi clip. Các chi phí này là dự tính, chưa đo.
- **Dữ liệu cần bổ sung:** tập cue có license, alignment và lineage kiểm được, ít nhất một nguồn generator/corpus held-out, cùng group REAL-only để kiểm tra FA. Không mở calibration hay official evaluation đã khóa trong repo.

## Thứ tự công việc

1. Dùng score đã seal để làm thống kê ghép nguồn theo từng order, đối chiếu gain và noise. Sau đó viết trước protocol kiểm soát cùng noise insertion, mức gain, delay và bitrate cho cohort độc lập.
2. Chuẩn bị replication hẹp hai seed 6207/6307 cho hai baseline. Chốt trước policy tính gradient, validation, checkpoint selection và B clean guardrail. Báo cáo từng seed, delta ghép và giới hạn số seed. Kiểm tra tài nguyên trước khi xin chạy, tạo ID mới nếu thay đổi config. Cặp 6107 có migration số học nên giữ như exploratory anchor.
3. Làm kiểm tra cue-target có giấy phép và matched lineage trước mọi pilot loss. Chỉ nâng C2 khi kiểm soát nuisance, giữ được B clean và vượt đối chứng trên speaker/generator chưa dùng. Foundation, calibration và cascade chờ quyết định mới.

C1 hiện là đóng góp thực nghiệm/engineering về order và lớp. C2 chưa xác lập novelty. C3 chờ policy và calibration độc lập. [Decision JSON](PHASE8_POST_TRAINING_DECISION.json)
