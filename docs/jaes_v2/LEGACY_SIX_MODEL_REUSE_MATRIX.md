# Audit sáu detector conference

Ngày 11/11/2026. Đối chiếu bản conference có SHA-256 `91fa8753025895bbf88470f4ebe7e6c283e897c8370fa8e91b1719abaac61073`, phản hồi reviewer, registry, adapter, metadata, artifact và các bảng benchmark. Hash cả sáu artifact đều khớp benchmark. Các cột kết quả mô tả những artifact đã đánh giá trong conference.

| Detector, framework và artifact | Nguồn weights, trạng thái checkpoint | Threshold | Vai trò và quyết định |
| --- | --- | ---: | --- |
| MobileNetV3Small–LSTM, TensorFlow. `models/lava_mobilenetv3_lstm.keras` | `VERIFIED`. ImageNet, huấn luyện local trên canonical split, 50 epoch warm-up và 50 fine-tune 20 layer cuối với BN frozen, best validation loss ở epoch 100. | 0,82 | `HISTORICAL_CONFERENCE_REFERENCE`, `ELIGIBLE_FOR_CONTROLLED_INFERENCE`. Dùng làm tham chiếu inference khi protocol phù hợp. Journal dùng bốn selected checkpoint riêng. `TRAINING_NOT_JUSTIFIED`. |
| ShuffleNetV2-1.0×–LSTM, TensorFlow. `models/shufflenetv2_lstm/model.keras` | `VERIFIED`. Scratch local, 28 epoch, best epoch 16, bản chuyển đổi có parity. | 0,12 | `HISTORICAL_CONFERENCE_REFERENCE`, `ELIGIBLE_FOR_ARCHITECTURE_TRANSFER` về kiến trúc. Checkpoint hiện hữu có thể phục vụ inference lịch sử. Kiểm tra transfer của hiệu ứng augmentation cần một cặp huấn luyện riêng với cùng protocol. `TRAINING_NOT_JUSTIFIED` cho V2. |
| MnasNet-A1-1.0–LSTM, TensorFlow. `models/mnasnet_lstm/model.keras` | `VERIFIED`. Scratch local, 39 epoch, best epoch 27, bản chuyển đổi có parity. | 0,90 | `HISTORICAL_CONFERENCE_REFERENCE`. Dùng artifact hiện hữu nếu cần tái kiểm tra inference lịch sử. `TRAINING_NOT_JUSTIFIED`. |
| EfficientNet-B0–LSTM, TensorFlow. `models/efficientnet_b0_lstm/model.keras` | `VERIFIED`. ImageNet, artifact warm-up được chọn tại epoch 47, có parity khi chuyển đổi. Fine-tuning tương đương là `NOT_EXECUTED` trong benchmark được báo cáo. | 0,90 | `HISTORICAL_CONFERENCE_REFERENCE`, `NOT_SCIENTIFICALLY_COMPARABLE` như kết quả fine-tune hoàn chỉnh. `TRAINING_NOT_JUSTIFIED`. |
| RawNet2, PyTorch → ONNX Runtime. `models/rawnet2_pretrained/model.onnx` | `PARTIALLY_VERIFIED`. Hash, adapter và ONNX parity đã kiểm. Metadata nhận diện baseline ASVspoof 2021 DF của Tak và cộng sự. Protocol 2021 quy định train/dev từ ASVspoof 2019. Danh sách source và lịch sử chọn checkpoint upstream cụ thể là `UNKNOWN`. Local train là `NOT_EXECUTED`. | 0,50 mặc định | `PRETRAINED_INFERENCE_REFERENCE`, `REQUIRES_PROVENANCE_AUDIT` trước khi dùng trên dữ liệu mới. Cùng hệ ASVspoof nên phải kiểm tra overlap upstream. `TRAINING_NOT_JUSTIFIED`. |
| AASIST, PyTorch → ONNX Runtime. `models/aasist_pretrained/model.onnx` | `PARTIALLY_VERIFIED`. Hash, adapter và ONNX parity đã kiểm. Metadata nhận diện pretrained NAVER. Repo upstream dùng ASVspoof 2019 LA cho train, validation và evaluation. Lịch sử chọn weights và exact source list upstream là `UNKNOWN`. Local train là `NOT_EXECUTED`. | 0,50 mặc định | `PRETRAINED_INFERENCE_REFERENCE`, `REQUIRES_PROVENANCE_AUDIT`. Dùng như artifact tham chiếu lịch sử. `TRAINING_NOT_JUSTIFIED`. |

## Định danh checkpoint

| Artifact | SHA-256 |
| --- | --- |
| MobileNetV3Small–LSTM | `7345bdeb9542adb7686e3cefbf070ad30f2b0de0da77ef5e1d644dd743763c84` |
| ShuffleNetV2–LSTM | `0a7dc43f6731e7c5c0379c2c9a341e7358a81bc37a7bedf4786e006942ec9442` |
| MnasNet–LSTM | `2c5301b2c3dc3e3faeb2e8ff11409c0b2dd637568a407c5a7b4eefba02055b02` |
| EfficientNet-B0–LSTM | `853a428ad0280faa644aaad0b4d3d6e25d9130b70b4167dfed9a7193a1b053f2` |
| RawNet2 ONNX | `afdaf27fa4e010b8a6d3051b37ea8d6ffb27dbaa7678faf22fef53285d515671` |
| AASIST ONNX | `7e7bbe8baf2dd8f6a09dcb5236133c81ea7c9a74b12929d685b39fb0026d776c` |

## Input, label và độ tương thích

- Bốn TF dùng sáu frame Mel theo thời gian, audio mono 22.050 Hz dài 3 s, segment 0,5 s, 128 Mel bands, ảnh 224×224×3. Backbone TimeDistributed → LSTM128 → Dense64 ReLU → dropout 0,4 → sigmoid. Convention local REAL=0, FAKE=1. Score sigmoid hướng FAKE. Threshold chọn theo F1 lớp FAKE trên 2.733 bản ghi validation legacy. Các adapter và artifact đã có bằng chứng restore/inference conference và parity cho các bản chuyển đổi. Journal còn dùng hợp đồng P0, checkpoint CLEAN_BCE và threshold 0,5 riêng.
- RawNet2 dùng waveform với Sinc front-end, residual, attention và GRU. AASIST dùng waveform với graph attention phổ–thời gian. Cả hai adapter ONNX dùng mono 16 kHz, 64.600 mẫu, prefix/zero padding xác định, tensor float32. Native FAKE=0, REAL=1, lấy softmax index 0 để xuất score hướng FAKE. Threshold 0,5 là mặc định chưa hiệu chuẩn trên validation legacy. Repo có cả registry PyTorch cho các trainable ID và registry ONNX pretrained. Hai artifact được conference chấm là bản ONNX.

## Hiệu năng đã lưu

| Detector | Clean F1 | Clean AUC | EER % | ΔF1 noise / codec / replay | CPU mean / P95 ms | RSS MiB |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| MobileNetV3Small–LSTM | 0,9724 | 0,9911 | 2,50 | 0,6722 / 0,0095 / 0,0620 | 43,81 / 52,12 | 554,55 |
| ShuffleNetV2–LSTM | 0,9824 | 0,9929 | 1,46 | 0,3555 / 0,0032 / 0,0256 | 62,52 / 75,37 | 501,82 |
| MnasNet–LSTM | 0,9645 | 0,9886 | 3,10 | 0,6119 / 0,0233 / 0,1489 | 117,00 / 138,96 | 566,04 |
| EfficientNet-B0–LSTM | 0,9563 | 0,9877 | 4,00 | 0,8053 / 0,0000 / 0,0641 | 175,85 / 196,11 | 651,23 |
| RawNet2 | 0,5330 | 0,5178 | 48,13 | 0,5077 / −0,0007 / −0,0109 | 96,54 / 102,62 | 249,50 |
| AASIST | 0,5139 | 0,5597 | 44,63 | −0,0660 / −0,0074 / −0,0682 | 362,91 / 398,45 | 442,54 |

Clean là 2.737 test recordings. Stress là 100 recordings cố định, bốn mức AWGN, bốn codec và replay mô phỏng. ΔF1 là clean trừ stressed trong diagnostic, với noise và codec tổng hợp qua các điều kiện của họ. Δ âm của external artifacts đi cùng clean baseline yếu. Full-test stress là `NOT_EXECUTED`. Timing là end-to-end batch 1 float32, một thread trên Ryzen 9 8940HX, 10 warm-up và 50 lần đo. Các số chỉ hỗ trợ so sánh artifact dưới protocol conference. So sánh trực tiếp với Journal V2 là `NOT_COMPARABLE` vì population, input pipeline, huấn luyện, ngưỡng và stress khác nhau.

## Audit dataset legacy

| Manifest | Tổng | REAL | FAKE |
| --- | ---: | ---: | ---: |
| Scanned | 18.722 | 10.550 | 8.172 |
| Retained | 18.232 | 10.493 | 7.739 |
| Train | 12.762 | 7.345 | 5.417 |
| Validation | 2.733 | 1.573 | 1.160 |
| Test | 2.737 | 1.575 | 1.162 |

Có 435 nhóm SHA-256 lặp, 476 bản sao dư nếu giữ một file mỗi nhóm. 14 nhóm có xung đột label chứa 30 file. Canonicalization loại tổng cộng 490 file và mỗi checksum retained chỉ xuất hiện một lần. ID mẫu là định danh manifest, còn speaker, source, generator, dataset và parent đều UNKNOWN ở cả 18.722 hàng. Đây là tập nội bộ đã tập hợp, nguồn corpus cụ thể chưa xác định. Có 0 checksum trùng với ba manifest ASVspoof foundation đã đối chiếu. Audio đã mã hóa lại, utterance trùng hoặc cùng speaker vẫn có thể nằm hai bên. Chỉ có căn cứ tái lập đánh giá legacy theo byte. Controlled training theo speaker hoặc generator đòi hỏi provenance mới trước khi cân nhắc trộn dữ liệu.

[Benchmark](../../models/lava_6_benchmark.json) · [Bảng chi phí](../../outputs/lava_6/lava_6_results.csv) · [Conference](../../papers/PaperID_804_final.docx) · [Phản hồi reviewer](../../papers/Response_to_Reviewers_final.docx) · [ASVspoof 2021](https://www.asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf) · [AASIST upstream](https://github.com/clovaai/aasist)
