# Báo cáo thu nhận dữ liệu R08

Đã xác minh metadata của ZIP In-the-Wild ở revision `eee168f92c367f8c82ff2cf42b6f61e362fd6211` bằng HTTP Range. Máy chủ trả 206, Content-Range và ETag đúng đối tượng. Đã so metadata SHA-256 đã ghim và đối chiếu đủ 31.779 WAV trong central directory. SHA-256 của toàn ZIP là giá trị công bố từ nhà phân phối. [Bằng chứng metadata](../../outputs/jaes_v2/r08_external_v1/metadata.json)

| Hạng mục | Quan sát |
| --- | --- |
| Bytes qua HTTP | 3.448.833 bytes ZIP directory và metadata. 0 byte WAV |
| Phân bố cấp file | 19.963 REAL, 11.816 FAKE, 54 nhãn speaker |
| Hỗ trợ quota | 41 nhóm nhãn đủ 16 file mỗi lớp. 49 nhóm đủ 8 mỗi lớp. Chưa xác lập số nguồn đủ điều kiện |
| WAV, P0 và lỗi dữ liệu | Decode và P0 pass trên WAV fixture tổng hợp. Chưa lấy WAV thật nên chưa có số clip thực lỗi, trùng hoặc đã qualified |
| Snapshot địa phương | 13/11/2026 18:47 UTC+07. RAM khả dụng 6,37 GiB, commit khả dụng 9,60 GiB, C trống 10,00 GiB, D trống 27,22 GiB. Preflight đạt |
| Kết quả acquisition / inference | Chưa có bộ 512 hoặc 256 clip. Chưa có protocol đã khóa và 0 external prediction |

## Nghiệm thu inference trên development

Đã kiểm cả sáu seal 640, xác minh checkpoint được chọn và hash trọng số sau restore. Trên ba fixture development cố định mỗi model (REAL A, FAKE A, REAL B), tensor P0 khớp chính xác với pipeline cũ. Cả 18 điểm khớp với điểm clean đã seal, sai khác lớn nhất 0, và lặp forward hai lần cho cùng điểm. Không dùng official evaluation để debug. Đã chạy chín test fixture và thử CLI scoring thiếu phê duyệt, bị chặn trước khi mở audio. Audit cuối kiểm đủ tám cây seal, 1.088 shard held-out gốc và hash V7. [Replay](../../outputs/jaes_v2/r08_external_v1/inference_qualification.json) · [Integrity](../../outputs/jaes_v2/r08_external_v1/protected_after.json)

Runner kiểm CRC và khóa member an toàn. Nó ghi ngân sách yêu cầu trước mỗi HTTP Range, để một lần ngắt không đặt lại ngân sách. WAV hoàn tất được giữ theo CRC khi resume, tiếp tục kiểm decode, hash và P0. File invalid, trùng hoặc overlap làm dừng quy trình và lưu lý do. Quy tắc fallback chỉ áp dụng thiếu quota cấp metadata, trước decode và trước scoring. Không thay clip sau khi xem kết quả. Phương án ZIP đầy đủ cần phê duyệt riêng và kiểm SHA-256 toàn file. [Hướng dẫn thao tác](R08_AUTHOR_EXECUTION_GUIDE.md)
