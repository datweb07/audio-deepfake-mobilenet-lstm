# Bàn giao held-out và manuscript V2

Ngày 11/11/2026. Trạng thái **HELDOUT_COMPLETED_MANUSCRIPT_V2_READY**.

## Thực thi và kiểm tra

Đạt đã phê duyệt scoring rõ ràng trong chat ngày 11/11/2026 và yêu cầu Codex điền hồ sơ. File phê duyệt ghi tên Đạt, timestamp có múi giờ và tham chiếu đến xác nhận đó. Ba hash proposed lock, selected rows và license khớp. Bốn selected-model export khớp danh tính trong lock. Runner hiện có đã chạy thành công một lượt, với mutex của repository và preflight PASS. Thời gian runner ghi nhận là 2.292,71 giây, khoảng 38,2 phút, gồm kiểm tra và scoring.

Đủ 1.088 recording, 67 speaker, 4 model, 4 điều kiện và 17.408 prediction. Kiểm tra 1.088 shard cho thấy đủ lưới source–condition–model, score hữu hạn trong [0,1], metadata đúng manifest và decision đúng ngưỡng 0,5. Phân tích tái tính khớp hoàn toàn metrics và bootstrap của runner. Mỗi endpoint và attack contrast có 10.000 draw hợp lệ, 0 không hợp lệ. Mỗi draw giữ nguyên các recording, view và cặp model thuộc speaker được lấy mẫu.

Preflight đầu lượt đạt với 5,45 GiB RAM, 9,78 GiB commit khả dụng, C 9,63 GiB và D 19,56 GiB. Snapshot sau scoring và build vẫn PASS, với 6,10 GiB RAM, 9,07 GiB commit, C 8,33 GiB và D 19,54 GiB. Snapshot sau được lưu trong thư mục analysis. Dữ liệu theo từng thời điểm không đo peak của tiến trình scoring.

Hai nhóm test audit và empirical execution đạt 14/14. Smoke check bằng dữ liệu tổng hợp kiểm tra xuất báo cáo, hình và bảng. Test tạm được chuyển vào đường dẫn ngắn trong workspace sau khi đường dẫn TEMP mặc định vượt giới hạn Windows. Toàn bộ bằng chứng thực nghiệm đến từ scoring đã hoàn thành.

## Kết quả chính

| Held-out endpoint | 9107 ERM | 9107 Channel | 9207 ERM | 9207 Channel |
| --- | ---: | ---: | ---: | ---: |
| Macro stress ROC-AUC | 0,8308 | 0,8875 | 0,7498 | 0,8948 |
| Stress FAKE miss | 1.370/2.496 | 587/2.496 | 815/2.496 | 453/2.496 |
| Stress REAL false alarm | 80/768 | 81/768 | 322/768 | 136/768 |
| Clean ROC-AUC | 0,9534 | 0,9436 | 0,9507 | 0,9410 |
| Clean FAKE miss | 125/832 | 129/832 | 127/832 | 147/832 |
| Clean REAL false alarm | 25/256 | 19/256 | 22/256 | 16/256 |

Stress ranking và pooled FAKE sensitivity cải thiện trong cả hai seed. Clean FAKE misses tăng, clean AUC giảm, và A17–A19 có thêm stressed misses trong cả hai seed. Những failure theo attack tiếp tục tồn tại dưới mức cải thiện tổng hợp. Độ lớn thay đổi giữa development và held-out phụ thuộc endpoint và quần thể. Development B có REAL-only speaker stratum riêng, trong khi held-out dùng quota và attack set khác.

Order asymmetry theo lớp tiếp tục xuất hiện trên cùng recording. Noise-to-codec gây nhiều REAL-only errors hơn, còn codec-to-noise gây nhiều FAKE-only misses hơn. Channel làm giảm nhiều exclusive errors, nhưng kết quả vẫn phụ thuộc attack và seed. Cơ chế forensic độc lập, transfer sang corpus khác và biến thiên trên quần thể initialization còn chưa được xác lập.

## Manuscript V2

Bản một cột có **30 trang A4, khoảng 8.565 từ nội dung chính, 13 hình vector, 9 bảng và 26 nguồn trích dẫn**. Abstract có 163 từ. Bốn hình mới trình bày contrast development/held-out kèm khoảng bất định, ROC held-out, lỗi A07–A19 và paired order scores. Hai bảng mới trình bày endpoint và điều kiện held-out.

Các mục được cập nhật gồm Introduction, Related Work, Experimental Setup, Results, Discussion, Limitations, Conclusion, abstract và claim map. Source phân tích mới ở trong thư mục manuscript. Bản JAES hai cột được build kiểm tra tương thích và có 21 trang trong scratch build. Hai PDF lịch sử vẫn khớp hash baseline. Bản PDF mới được xuất riêng với hậu tố v2.

Hai layout build thành công, với citations và cross-references phân giải đầy đủ. Không có overfull box hoặc ký tự thiếu trong log. Đã render và rà toàn bộ 30 trang review, bao gồm bảng rộng, biểu đồ, disclosure và bibliography.

## File và thư mục gửi AI

Từ root repository, gửi những mục sau, giữ cấu trúc thư mục nếu đóng ZIP.

1. `papers/jaes_v2/jaes_manuscript/build/LAVA_JAES_Review_Draft_v2.pdf`.
2. `outputs/jaes_v2/phase10/empirical_execution_v1/heldout_28f940ea7d151177/metrics.json`.
3. Toàn bộ thư mục `analysis/` trong run held-out. Thư mục này chứa `scores.csv`, `condition_metrics.csv`, `attack_metrics.csv`, `verified_analysis.json`, `HELDOUT_SCIENTIFIC_REPORT.md`, snapshot tài nguyên, baseline manuscript và hồ sơ completion.
4. `manifest.json` của run held-out, proposed lock và `heldout_scoring_author_approval.json` tại thư mục empirical execution.
5. `papers/jaes_v2/jaes_manuscript/claim_evidence_map.csv` và báo cáo bàn giao này.

Nếu người nhận cần tự kiểm tra hash từng shard, gửi thêm thư mục `scores/` của run held-out. Nếu cần sửa LaTeX, gửi hai entry point, shared TeX, `sections/`, `tables/`, `figures/`, bibliography, class và các style/support files hiện có. README hướng dẫn chọn `main_review.tex` trên Overleaf với pdfLaTeX.

## Việc tác giả cần làm trước 20/12

- Gửi PDF V2 và báo cáo khoa học cho giảng viên. Xin góp ý về cách diễn giải empirical contribution và các kết quả bất lợi.
- Xác nhận thông tin tác giả, affiliation, corresponding author, biography và disclosure cuối với các đồng tác giả.
- Cung cấp venue, tình trạng proceedings, DOI nếu đã có, cùng thỏa thuận xuất bản của conference để hoàn thiện reference và kiểm tra quyền mở rộng.
- Tích hợp góp ý trong tháng 11 đến đầu tháng 12, rồi rà claim, nguồn trích dẫn, hình/bảng và bộ source có thể build lại trong giai đoạn 11–14/12.
- Hoàn thiện metadata, cover letter và gói nộp trong giai đoạn 15–20/12.

Các quyết định nghiên cứu hiện tại tiếp tục là seed 9307 NO-GO, medium-plus NO-GO, foundation NO-GO cho deadline tháng 12, C2 rejected và C3 deferred. Ba contribution trong bài diễn đạt các đóng góp thực nghiệm. Bản V2 đủ để giảng viên phản hồi, còn mức sẵn sàng nộp phụ thuộc việc hoàn tất phản biện nội bộ, metadata, reference conference và quyền xuất bản.
