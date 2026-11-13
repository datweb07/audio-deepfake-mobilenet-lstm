# Kiểm tra máy và quyết định R08

**Ngày 13/11/2026. Quyết định cho V7 và hạn 20/12/2026 là NO-GO cho external scoring.** R08 vẫn mở về thực nghiệm. Giữ giới hạn external validity đã ghi và xin giảng viên xác nhận phạm vi. Tôi đánh giá **R08 đáng làm ở vòng sau**, khi có tập ngoài corpus đủ căn cứ. Thử nghiệm này trả lời trực tiếp liệu lợi ích xếp hạng và sự đánh đổi giữa REAL false alarm và FAKE miss có lặp lại trên nguồn khác hay không. Nếu giảng viên yêu cầu kết quả ngay trong vòng sửa này, ưu tiên tháo vướng về dữ liệu trước khi cam kết V8.

## Cấu hình và tài nguyên

Snapshot từ lệnh `phase10_experiment resources` lúc 17:57:53, UTC+07. Đây là phép đo tức thời.

| Hạng mục | Đo được | Ý nghĩa đối với R08 |
| --- | --- | --- |
| Hệ điều hành và CPU | Windows build 26300, AMD Ryzen 5 7535HS, 12 bộ xử lý logic | Giữ môi trường CPU đã dùng cho nghiên cứu |
| RAM | 15,32 GiB hệ thống nhận, 3,38 GiB khả dụng | Đạt sàn 2 GiB tại thời điểm đo |
| System commit | 11,54 GiB còn khả dụng | Đạt sàn 3 GiB |
| Ổ C | 9,22 GiB trống | Đạt sàn 3 GiB |
| Ổ D | 27,22 GiB trống. Còn 12,10 GiB (12,99 GB) sau dự phòng 15 GiB + 128 MiB | Đủ cho phương án ZIP đơn lẻ hoặc lấy tập con có giới hạn |
| GPU | RTX 3050 Laptop, tổng 4.096 MiB VRAM, còn 3.718 MiB khi đo, driver 595.71 | Driver thấy GPU. TensorFlow 2.15.0 trong venv chỉ liệt kê CPU và `is_built_with_cuda()` trả `False` |
| Runtime | Python 3.11.9, TensorFlow/Keras 2.15.0 | `RESOURCE_PREFLIGHT_PASS`. Kết luận khả thi dựa trên CPU và benchmark development đã có |

TensorFlow 2.10 là bản cuối hỗ trợ CUDA trên Windows native theo tài liệu chính thức. Phương án R08 giữ pipeline CPU, chạy batch 1 và nạp từng checkpoint. Sáu file `selected.keras` của ba seed 640 đã được đối chiếu SHA-256 với seal riêng. Chưa đo thông lượng trên audio In-the-Wild. [Hướng dẫn TensorFlow](https://www.tensorflow.org/install/pip) · [Benchmark cũ](../../outputs/jaes_v2/supervisor_major_revision/components_9207/efficiency.json)

## Dung lượng và điều kiện của từng phương án

| Phương án | Đánh giá |
| --- | --- |
| ASVspoof 2021 DF đầy đủ | 34,5 GB nén (~32,13 GiB), vượt 27,22 GiB trống của D ngay cả trước khi giải nén. Metadata còn cho thấy 4.455 trial DF có parent VCTK đã biết trùng với tập held-out cũ. [Phát hành chính thức](https://zenodo.org/records/4835108) · [Rà overlap](../../outputs/jaes_v2/final_supervisor_closure/df_metadata_recheck.json) |
| In-the-Wild ZIP riêng | ZIP 8,16 GB (~7,60 GiB). Nếu lưu trên D, còn khoảng 4,50 GiB trên mức dự phòng. Có thể đọc từng member hoặc lấy tập con bằng HTTP Range. Probe cũ mới đọc metadata và danh mục member. [Archive ghim phiên bản](https://huggingface.co/datasets/mueller91/In-The-Wild/blob/eee168f92c367f8c82ff2cf42b6f61e362fd6211/release_in_the_wild.zip) · [Probe metadata](../../outputs/jaes_v2/final_supervisor_closure/itw_metadata_probe.json) |
| In-the-Wild ZIP và giải nén toàn bộ | Khoảng 16,89 GB. Thiếu 3,89 GB (~3,63 GiB) so với ngân sách sau dự phòng, trước khi tính file tạm. |

**Điểm quyết định là dữ liệu.** Trang Fraunhofer ghi Apache 2.0, còn bản phân phối đã ghim của tác giả ghi CC BY-SA 4.0. Hai nhãn này cần được làm rõ theo đúng bản phân phối và audio sẽ dùng. Metadata hiện có filename, speaker và label. Cần đối chiếu danh tính speaker, nguồn video hoặc parent, trùng nội dung và quan hệ giữa các đoạn cắt. Bài gốc xác nhận đây là corpus ngoài miền ASVspoof; điều đó chưa tự xác nhận độc lập cho từng clip được chọn trong nghiên cứu này. [Fraunhofer](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild) · [Bản phát hành của tác giả](https://huggingface.co/datasets/mueller91/In-The-Wild/blob/eee168f92c367f8c82ff2cf42b6f61e362fd6211/release_in_the_wild.zip) · [Bài Interspeech](https://www.isca-archive.org/interspeech_2022/muller22_interspeech.pdf)

## Phương án khi mở lại R08

1. Xác nhận điều khoản sử dụng của release và audio, kiểm tra đủ 16 speaker có hỗ trợ cả hai lớp ở cấp nguồn. Nếu đủ, chọn 16 REAL và 16 FAKE nguồn duy nhất mỗi speaker bằng seed 20261112. Đối chiếu với train, development, calibration và held-out cũ theo speaker/alias, ID nguồn và kiểm tra trùng audio theo khả năng dữ liệu. Nếu thiếu quota, ghi lại và chốt lại thiết kế trước scoring.
2. Sau khi xác lập quyền sử dụng, lấy một đợt nhỏ để kiểm tra tải và decode thực tế. Khóa danh sách 512 mẫu, hash audio, nhóm speaker và nguồn, sáu checkpoint 640, preprocessing P0, một view native, hướng điểm P(FAKE) và ngưỡng 0,5. Nếu chỉ xác lập được khác biệt miền và các kiểm tra overlap khả thi, giới hạn kết luận ở **cross-domain diagnostic** và trình bày phần lineage còn thiếu. Nếu có đủ bằng chứng độc lập nguồn và speaker, có thể gọi là external validation theo phạm vi đã khóa.
3. Chỉ scoring sau khi tập dữ liệu qua audit và giao thức được chấp thuận. Dự kiến 3.072 dự đoán. Dùng ROC-AUC, REAL false alarm và FAKE miss trên cùng mẫu cho mọi checkpoint. Báo cáo riêng từng seed và độ bất định ghép cặp theo speaker, có xét quan hệ nguồn. Mốc khoảng 5,2 phút chỉ là ngoại suy decode/preprocessing và forward từ benchmark clean trên development. Thời gian tải, audit nguồn, decode validation, restore và xác minh sẽ tính riêng. Thiết kế một view native đo chuyển miền ở view native. Phạm vi robustness với noise/Opus ngoài corpus cần một thiết kế nhiều điều kiện.

**Việc làm ngay** là gửi báo cáo này cùng V7 cho giảng viên. Nếu giảng viên chấp nhận giới hạn, giữ R08 cho vòng sau. Nếu yêu cầu R08 trước khi nộp, thống nhất trước liệu một cross-domain diagnostic có đáp ứng nhận xét hay cần độc lập đến cấp nguồn. Chỉ đưa kết quả vào V8 sau scoring hợp lệ. Lượt này chỉ đo cấu hình, kiểm tra tài liệu và hash. Không tải audio, train hay chạy inference.

## Xác minh bản V7 đã gửi

PDF V7 có SHA-256 `735c8b1c8282ced03e9ac74f96c575f09566b3eab39b22e8dde5dad320e40732`. ZIP nguồn Overleaf có SHA-256 `b853652767d4ed5388ae9e5327a528722cf374de35ff7582e7411fb062a80214`. Cả hai khớp bản xác minh sau khi khôi phục lời cảm ơn UEH ở trang 31. JSON xác minh khoa học trước đó ghi hash của bản build cũ. Nhóm tác giả vẫn cần xác nhận phạm vi hỗ trợ UEH trước khi nộp journal. [Xác minh bản hiện hành](../../outputs/jaes_v2/phase10/jaes_build_work/v7_ack/acknowledgement_verification.json) · [Audit R01–R12](SUPERVISOR_12_REQUIREMENTS_FINAL_AUDIT.md)
