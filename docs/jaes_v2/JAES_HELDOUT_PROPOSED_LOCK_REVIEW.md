# Rà proposed lock held-out JAES

10/11/2026. `READY_FOR_PROPOSED_LOCK_REVIEW`. Phê duyệt audit được ghi theo xác nhận trong chat của Đạt. Preflight ngay trước audit PASS với 4,42 GiB RAM, 9,01 GiB commit, C 10,94 GiB và D 20,03 GiB. Auditor đọc CM protocol và license, đối chiếu ZIP và tính SHA-256 byte cho các FLAC được chọn. Không decode, inference, scoring, tính chỉ số hoặc train. Bản thảo giữ nguyên.

| Kiểm tra | Kết quả |
| --- | --- |
| Official CM trials | 7.355 REAL + 63.882 FAKE = 71.237 |
| Audio ZIP | 71.933 FLAC, 696 ngoài danh sách trial, 0 trial thiếu hoặc trùng ZIP member |
| Primary | 1.088 nguồn, 256 REAL + 832 FAKE, đủ 64 nguồn cho từng A07–A19, không thiếu quota |
| Speaker | 67 speaker, 3–25 nguồn mỗi speaker, 67 speaker REAL và 48 speaker FAKE |
| Trùng trong danh sách chọn | 0 sample ID, 0 SHA-256 byte |
| Overlap train, dev, calibration | 0 theo từng tiêu chí sample ID, speaker ID và SHA-256 byte với mỗi manifest |
| Lineage | CM protocol không đủ trường để xác nhận lineage generator hoặc utterance |
| Thiết kế | Bốn model và bốn điều kiện giữ nguyên, seed lấy mẫu 91109, threshold 0,5 với score FAKE, 17.408 prediction dự kiến |
| License | Open Data Commons Attribution License. Văn bản trong lock nêu điều kiện ghi công và phân biệt quyền cơ sở dữ liệu với quyền đối với từng nội dung. Không suy ra quyền phân phối audio gốc. |
| Trạng thái | `authorized: false`, `HELDOUT_NOT_AUTHORIZED` |

- Prelock SHA-256 `3ae003d298fd812196a3ae0b8e433f8061518805737264fb35cbd57a3465a58f`.
- Proposed lock SHA-256 `f422a0a20580bdca4c4a99c6aea58add2958edf3e7efabb50f57c274496f23a8`.
- Selected rows SHA-256, đã tái tính từ rows, `af2a8c73311cc051498532e35d7159c586f8b0387740b497d01464914cbec38f`.
- Official CM protocol SHA-256 `84fce576efff092269a0a6ad247d2e9f2f7b4d4e25a4545efb2518da92708036`.
- License SHA-256 `99a4d3e0db6f9a5cd8871f9737b207af01de333f57c6269a11ce755227c1555d`.

14 test liên quan đã đạt. Kiểm tra riêng 68 binding của prelock trước khi mở ZIP. Auditor sau phê duyệt kiểm tra thêm archive identity. Không sửa trainer, model, checkpoint hay bản conference.

## NOT_AUTHORIZED_FOR_EXECUTION

Lệnh sau chỉ dành cho giai đoạn sau khi tác giả rà license và danh sách mẫu, cấp phê duyệt scoring riêng gắn với ba hash proposed lock, selected rows và license, đồng thời preflight mới PASS.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.jaes_v2.empirical_heldout --proposed-lock outputs/jaes_v2/phase10/empirical_execution_v1/heldout_proposed_9107_9207_v1.json --scoring-approval outputs/jaes_v2/phase10/empirical_execution_v1/heldout_scoring_author_approval.json
```

Bước tiếp theo là rà proposed lock, duyệt scoring riêng, thực thi đánh giá khóa, phân tích kết quả, cập nhật và build bản thảo, xác nhận metadata và quyền conference trước 20/12. [Proposed lock](../../outputs/jaes_v2/phase10/empirical_execution_v1/heldout_proposed_9107_9207_v1.json) · [Handoff](JAES_FINAL_PRE_SUBMISSION_HANDOFF.md)
