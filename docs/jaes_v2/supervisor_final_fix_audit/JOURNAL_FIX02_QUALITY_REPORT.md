# JOURNAL FIX02 — kiểm tra chất lượng

## Nguồn

- FIX01 có 33 trang, A4 khoảng 595,276 × 841,890 pt, một cột, có lớp văn bản. SHA-256 `482e122bed05498673d50a89c18a3b78d59dce69cba321b0e0f444531852ae89`. Bản gốc và snapshot `outputs/jaes_v2/supervisor_final_fix/JOURNAL FIX01.pdf` cùng hash.
- LaTeX hiện tại lệch FIX01. `shared_metadata.tex` mở Abstract bằng “We examine how channel augmentation…” trong khi FIX01 mở bằng “The effect of augmentations…”. Section 6.2 trong `sections/results.tex` mô tả vượt guard, còn FIX01 ghi “passes”. Table 1, phương pháp và kết luận cũng có cách diễn đạt khác. Đối chiếu normalized page text với V7 cho 0/33 trang khớp hoàn toàn. Chưa tìm thấy nguồn LaTeX tái tạo FIX01.
- FIX02 lấy trực tiếp từ FIX01. Các toán tử PDF in văn bản cũ được gỡ khỏi content stream ở đúng baseline. Văn bản Latin Modern mới được chèn dạng ký tự có thể tìm kiếm và trích xuất. Logic kiểm tra đối chiếu từng vùng với câu thay thế. Tệp nguồn V7 vẫn là nguồn lịch sử. Khi cần biên dịch submission, phải có nguồn biên tập tương ứng để áp 13 thay đổi.

## Kiểm tra nội dung và hình thức

- FIX02 có 33 trang A4 một cột, 14 hình, 11 bảng và 26 tài liệu tham khảo. SHA-256 `64ea8a0d2a71b14e073d84c7e96a201799b74a38b08cbd28ce0e36fa0f1e1c52`.
- 13 vùng văn bản sửa trên 12 trang. So sánh text ngoài vùng sửa, ảnh nhúng, số đối tượng vector và bản render ngoài vùng sửa một cách tự động. Các trang không sửa khớp pixel ở độ phân giải kiểm tra. Giữ Table 1, Figure 12, các phương trình, footer, references, và nguyên câu cảm ơn UEH. Rà các trang trọng điểm bằng bản render, gồm Abstract, Introduction, bảng, Results, Discussion, Limitations, Conclusion và References. Kiểm tra nội dung trước/sau cho từng vùng trong [changelog](SUPERVISOR_FIX_CHANGELOG.md).
- Đếm 18 DOI trong PDF và đối chiếu với `reference_audit.csv`, gồm hai mã tách dòng. Đối chiếu này xác nhận chuỗi DOI và hồ sơ thư mục có sẵn. Việc xác minh trực tuyến từng DOI và quyền conference vẫn dành cho hồ sơ submission.
- FIX01 có mục Reproducibility và Acknowledgement. Data Availability và Code Availability riêng được chuẩn bị trong [checklist](JAES_FINAL_SUBMISSION_CHECKLIST.md) để chốt theo quyền chia sẻ thực tế. Chưa tuyên bố phát hành toàn bộ artifact.

## Đối chiếu số liệu

Tính lại ROC-AUC và lỗi tại ngưỡng 0.5 từ tám CSV development (2.576 score mỗi model) và CSV held-out gốc (17.408 score, 1.088 nguồn). Nội dung đối chiếu gồm từng điều kiện, clean A và B, pooled stress views, A05/A06 và A07–A19. B chỉ có REAL nên không tính AUC hai lớp. Interval và guard đối chiếu với JSON đã niêm phong, không thay quy trình bootstrap. Figure 12 khớp chiều và support. Các interval giữ theo `heldout_28f940ea7d151177/metrics.json` gốc. Bảng chẩn đoán hậu kiểm khác có một số endpoint interval khác nhẹ, cùng kết luận dấu và quan hệ với 0.

| Đối chiếu | Kết quả |
| --- | --- |
| Macro stress AUC development 640 | ERM→Channel 0,6715→0,8026 (9107), 0,5946→0,7839 (9207), 0,5498→0,7934 (9307). Ở 1.280 seed 9207 là 0,7062→0,8143. |
| 1.280 seed 9207 | Stress A REAL 5→161/336. Stress A FAKE 248→64/336. Clean A FAKE 3→8/112 và B REAL 18→22/144, đều FAIL. Selected tags ERM `u001600_e020_c000`, Channel `u001280_e016_c000`. |
| Held-out ranking | 0,8308→0,8875 (9107), 0,7498→0,8948 (9207). Clean AUC giảm ở cả hai. |
| Held-out pooled stress | REAL 80→81/768 (9107) và 322→136/768 (9207). FAKE 1.370→587/2.496 và 815→453/2.496. |
| A05/A06 | Chênh lệch miss ở 640 lần lượt là (+2,+4), (+9,+15), (−6,−17) trên 54 views mỗi attack ở 9107, 9207, 9307. Ở 1.280 seed 9207 là (−14,−39). |
| A17–A19 | 9107 thêm (+10,+11,+4), 9207 thêm (+43,+43,+56), mẫu số 192 views mỗi attack. 9107 intervals chứa 0. 9207 intervals loại 0, pointwise. |
| Order | So sánh hai thứ tự trên cùng nguồn. Với A REAL seed 9207, contrast order chuyển từ −28,57 lên +26,79 điểm ở hai protocol. Mô tả detector response với cơ chế forensic chưa xác định. |

Nguồn bằng chứng gồm `outputs/jaes_v2/phase10/size1280_v1/size_sensitivity.json`, các CSV cùng seal development đã lưu và `outputs/jaes_v2/phase10/empirical_execution_v1/heldout_28f940ea7d151177/metrics.json` cùng `analysis/scores.csv`. Audit không inference, không tạo score hay checkpoint mới.
