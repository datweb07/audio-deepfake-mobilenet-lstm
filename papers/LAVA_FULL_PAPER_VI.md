# LAVA: Khung Đánh giá Gọn nhẹ cho Phát hiện Giọng nói Deepfake Bền vững và Thời gian Thực

## Tóm tắt

Các detector giọng nói deepfake thường được so sánh trên các phân hoạch dữ liệu, quy ước điểm số, tiền xử lý và giao thức runtime khác nhau. LAVA tạo một biên đánh giá chung nhưng bảo toàn kiến trúc native. Nghiên cứu hiện tại đánh giá sáu artifact: bốn detector chuỗi Mel do LAVA huấn luyện—MobileNetV3Small-LSTM, ShuffleNetV2-1.0x-LSTM, EfficientNet-B0-LSTM và MnasNet-A1-LSTM—cùng RawNet2 và AASIST pretrained bên ngoài. Clean evaluation dùng toàn bộ 2.737 bản ghi test. ShuffleNet đạt kết quả clean tốt nhất (F1 0,9824; AUC 0,9929; EER 1,46%), còn MobileNet có độ trễ thấp nhất (43,8 ms); ShuffleNet đạt 62,5 ms và RTF 0,0208 trên cùng CPU desktop một luồng. Robustness chỉ được đo ở mức chẩn đoán trên tập con phân tầng cố định 100 bản ghi. Mean ΔF1 của ShuffleNet qua chín condition là 0,1623; nhiễu SNR thấp vẫn là điểm yếu chính. Pareto chẩn đoán gồm MobileNet, ShuffleNet và AASIST. Khác biệt provenance và input contract không cho phép diễn giải đây là so sánh thuần kiến trúc; physical replay, unseen dataset, edge device, đa seed và full-test robustness vẫn chưa có.

**Từ khóa—** phát hiện giọng nói deepfake; chống giả mạo âm thanh; học sâu gọn nhẹ; độ bền vững; real-time factor; phân tích Pareto.

## 1. Giới thiệu

### 1.1 Bối cảnh và động lực

Giọng nói tổng hợp và chuyển đổi đặt ra rủi ro cho người nghe lẫn hệ thống xác thực người nói. ASVspoof đã chuẩn hóa các tình huống logical access và physical access [1–5], còn WaveFake mở rộng nguồn âm thanh sinh công khai [6]. Các detector hiện dùng nhiều hướng: đặc trưng âm học, waveform thô, graph attention và backbone tích chập gọn nhẹ. Tuy nhiên, chất lượng dự đoán đơn lẻ không cho biết detector có bền vững trước biến dạng kênh hay có phù hợp với giới hạn runtime hay không.

So sánh dễ sai lệch khi các hệ thống đảo thứ tự lớp, trả logit thay vì xác suất có cùng nghĩa, dùng thời lượng đầu vào khác nhau, hoặc đo riêng neural forward ở mô hình này nhưng đo toàn pipeline ở mô hình khác. Dữ liệu trùng lặp tạo rủi ro thứ hai: các bản âm thanh giống hệt theo byte nằm ở các split khác nhau có thể làm tăng ảo hiệu năng held-out. LAVA chuẩn hóa biên đánh giá thay vì ép mọi detector vào cùng một kiến trúc.

### 1.2 Khoảng trống nghiên cứu

RawNet2 xử lý trực tiếp waveform [9], AASIST duy trì tương tác graph phổ–thời gian [10], còn các CNN di động được thiết kế theo các nguyên lý hiệu quả khác nhau [20–25]. Ép các hệ thống này vào một topology sẽ làm mất tính đa dạng có ý nghĩa. Ngược lại, đánh giá không có hợp đồng chung sẽ trộn lẫn tác động của kiến trúc, semantics điểm số, dữ liệu và cách đo.

Repository hiện thực hóa biên chung đó cho bốn artifact lightweight có lịch sử khởi tạo khác nhau và hai artifact tham chiếu được huấn luyện từ bên ngoài. Clean đã hoàn tất cho cả sáu; robustness mới hoàn tất trên tập con chẩn đoán cố định.

### 1.3 Câu hỏi nghiên cứu

**RQ1:** Các detector CNN thời gian gọn nhẹ cạnh tranh như thế nào với hai hệ thống chống giả mạo pretrained bên ngoài trong giao thức clean chung của LAVA?

**RQ2:** Hiệu năng thay đổi như thế nào dưới các điều kiện noise, codec và simulated replay đã thực sự được chạy?

**RQ3:** Detector nào không bị trội hoàn toàn khi đồng thời xét EER, suy giảm độ bền vững trung bình và RTF end-to-end trong phạm vi chẩn đoán đã hoàn tất?

### 1.4 Đóng góp

Bài báo đóng góp: (1) hợp đồng `P(FAKE)` độc lập framework; (2) giao thức toàn vẹn dữ liệu dựa trên checksum; (3) benchmark clean sáu detector và robustness diagnostic có thể truy vết; (4) giao thức CPU chung; và (5) phân tích lỗi, bootstrap, agreement và Pareto từ số đo thật.

## 2. Công trình liên quan

### 2.1 Benchmark deepfake âm thanh và chống giả mạo

ASVspoof qua các edition 2015–2021 bao quát synthesis, conversion, replay và deepfake, đồng thời dùng EER/t-DCF theo từng track [1–5]. WaveFake tập hợp âm thanh từ nhiều họ mô hình sinh và ngôn ngữ [6]. Chúng tạo bối cảnh nghiên cứu, nhưng dữ liệu LAVA hiện tại không được tuyên bố là tái tạo các protocol này.

### 2.2 Mô hình waveform thô

RawNet2 được áp dụng cho chống giả mạo theo hướng end-to-end [9]. LAVA giữ input waveform, front end kiểu Sinc, khối residual thời gian, attention, GRU và classifier. Trọng số hiện tại là checkpoint bên ngoài, không phải artifact được LAVA huấn luyện lại.

### 2.3 Detector graph phổ–thời gian

AASIST tích hợp graph phổ và thời gian bằng heterogeneous graph attention, graph pooling, master/stack node và readout mở rộng [10,27]. LAVA chỉ thích nghi loader và score semantics, không biến AASIST thành Mel-CNN-LSTM.

### 2.4 Kiến trúc tích chập gọn nhẹ

MobileNetV3 kết hợp hardware-aware search với cải tiến kiến trúc [20,21]; EfficientNet scale đồng thời depth, width và resolution [24]; MnasNet đưa latency vào mục tiêu architecture search [25]; ShuffleNetV2 nhấn mạnh các nguyên tắc tốc độ thực tế [22,23]. LAVA dùng cả bốn làm embedding theo segment. ShuffleNet được huấn luyện scratch end-to-end và hiện đã tham gia benchmark.

### 2.5 Robustness và đánh giá hướng triển khai

Noise, codec và replay có thể phá vỡ các dấu hiệu mà detector dựa vào. Giao thức hợp lệ cần sinh mỗi file stress một lần, dùng lại cho mọi model và so với đúng clean sample. FLOPs cũng không thay thế latency thực. Vì vậy LAVA đo kích thước artifact, RSS tiến trình, preprocessing, model-only, end-to-end, throughput và RTF. Kết quả hiện tại là bằng chứng CPU desktop, không phải thiết bị di động hay edge.

### 2.6 Biểu diễn tự giám sát và khả năng tổng quát hóa

wav2vec 2.0, HuBERT và WavLM cung cấp biểu diễn tiếng nói tự giám sát có khả năng chuyển giao [28–30]; việc fine-tune wav2vec với augmentation đã cho kết quả chống giả mạo mạnh [12]. Tuy nhiên, các nghiên cứu đánh giá đồng nhất và cross-domain chỉ ra rằng hiệu năng in-domain cao không bảo đảm tổng quát hóa [13–19]. LAVA hiện chưa có experiment trên corpus ngoài, do đó không biến năng lực importer tương lai thành kết quả đã đo.

### 2.7 Kiến trúc gọn nhẹ và đo lường triển khai

MobileNetV2/V3 [20,21], ShuffleNet/V2 [22,23], EfficientNet [24] và MnasNet [25] đại diện cho các chiến lược khác nhau về inverted residual, depthwise convolution, channel shuffle, compound scaling và tìm kiếm kiến trúc theo latency. LAVA chỉ dùng chúng làm encoder theo segment; số liệu ImageNet hoặc điện thoại trong paper gốc không phải số liệu deepfake audio của LAVA. Vì FLOPs, số tham số và thời gian thực thi không tương đương, framework đo trực tiếp end-to-end latency và RTF.

### 2.8 Thống kê và lựa chọn đa mục tiêu

Bất định trên test được ước lượng bằng bootstrap [31]. Chênh lệch correctness ghép cặp dùng McNemar exact [32] và hiệu chỉnh Holm cho 15 cặp [33]. Pareto giữ riêng ba mục tiêu thay vì gộp bằng trọng số tùy ý; một điểm không bị trội vì thế không đồng nghĩa “model tốt nhất” trong mọi ứng dụng.

**Bảng 1. Vị trí của một số hướng nghiên cứu liên quan.**

| Công trình | Model/tài nguyên | Robustness/cross-domain | Efficiency | Quan hệ với LAVA |
|---|---|---|---|---|
| ASVspoof [1–5] | benchmark cộng đồng | LA/PA/deepfake | không phải trọng tâm chính | nền tảng protocol và EER |
| WaveFake [6] | dataset đa generator | đa dạng dữ liệu | không | ứng viên external future work |
| RawNet2 [9] | raw waveform | anti-spoofing | hạn chế | reference checkpoint ngoài |
| AASIST [10] | graph phổ–thời gian | anti-spoofing | hạn chế | reference checkpoint ngoài |
| wav2vec anti-spoofing [12] | encoder tự giám sát | augmentation/generalization | chưa có CPU protocol chung | họ model tương lai |
| Müller và cộng sự [13,15] | phân tích đồng nhất/cross-domain | trọng tâm | một phần | hỗ trợ diễn giải thận trọng |
| CD-ADD/XMAD-Bench [17,19] | cross-domain dataset | trọng tâm | không đồng nhất với LAVA | external evaluation tương lai |
| LAVA | sáu detector dị thể | clean và diagnostic noise/codec/replay | size, RSS, latency, throughput, RTF | contract hợp nhất |

## 3. Phương pháp

### 3.1 Tổng quan LAVA

LAVA là framework benchmark, không phải detector thứ bảy. Một benchmark được biểu diễn bởi $\mathcal{B}=\{\mathcal{D},\mathcal{M},\mathcal{C},\mathcal{E},\mathcal{A}\}$, trong đó $\mathcal{D}$ là dataset và integrity protocol, $\mathcal{M}$ là tập detector, $\mathcal{C}$ là các condition đã thực thi, $\mathcal{E}$ là detection/resource metrics, và $\mathcal{A}$ là phân tích thống kê, agreement, lỗi và Pareto. Bốn tầng phần mềm tương ứng là data/integrity, model/adapter, benchmark và analysis. Detector specification khai báo framework, loại input, thời lượng, đường dẫn artifact và provenance. Benchmark khóa hash model, metadata, threshold, manifest và source inference trước khi chạy; báo cáo tính lại metrics từ score theo từng sample.

![Hình 1. Kiến trúc sáu detector và biên đánh giá thống nhất của LAVA.](figures/lava_6_model_overview.png)

Pipeline (Hình 2) kiểm tra manifest, load tuần tự từng artifact, đánh giá điểm không đổi, sinh stress audio dùng chung, đo runtime và tạo bảng/hình bằng chương trình.

![Hình 2. Pipeline benchmark LAVA có thể tái lập.](figures/lava_benchmark_pipeline.png)

### 3.2 Dataset và giao thức toàn vẹn

Dataset nội bộ nằm trong `data/REAL` và `data/FAKE`; speaker, source, generator, parent recording và dataset ID đều `UNKNOWN`. Vì vậy bài báo không tuyên bố speaker-, source-, generator-disjoint hay cross-dataset.

Inventory scan 18.722 file: 10.550 REAL và 8.172 FAKE. SHA-256 phát hiện 435 nhóm trùng, 476 file trùng dư thừa và 14 nhóm checksum khác nhãn gồm 30 file. Mọi thành viên khác nhãn bị cách ly ở manifest. Trong nhóm cùng nhãn, đường dẫn đứng đầu theo thứ tự từ điển được giữ làm đại diện. Manifest cuối gồm 18.232 bản ghi (10.493 REAL; 7.739 FAKE), chia deterministic seed 42 thành train 12.762, validation 2.733 và test 2.737. Claim hợp lệ duy nhất là **checksum-group-disjoint**. Manifest hash là `8b55591d58d3658b8cafe0e77b6ebdedbaa67be2e339730a7276fe9b10958df9`.

Với file $x_i$, đặt $h(x_i)=\operatorname{SHA256}(x_i)$ và $x_i\sim x_j\iff h(x_i)=h(x_j)$. Nếu $G_{train}$, $G_{val}$ và $G_{test}$ là các tập nhóm checksum, utility integrity xác nhận $G_{train}\cap G_{val}=G_{train}\cap G_{test}=G_{val}\cap G_{test}=\varnothing$. Bảo đảm này hẹp hơn speaker-disjoint vì manifest không có speaker ID.

![Hình 3. Quy trình xây dựng manifest chính tắc có kiểm soát checksum.](figures/dataset_integrity_pipeline.png)

**Bảng 2. Tóm tắt toàn vẹn dữ liệu.**

| Thuộc tính | Giá trị |
|---|---:|
| Scan / included / excluded | 18.722 / 18.232 / 490 |
| REAL / FAKE included | 10.493 / 7.739 |
| Nhóm trùng / file trùng dư thừa | 435 / 476 |
| Nhóm xung đột / file bị cách ly | 14 / 30 |
| Seed / claim | 42 / checksum-group-disjoint only |

**Bảng 3. Các split chính tắc.**

| Split | Số mẫu | Tỷ lệ |
|---|---:|---:|
| Train | 12.762 | 70,0% |
| Validation | 2.733 | 15,0% |
| Test | 2.737 | 15,0% |

### 3.3 Tiền xử lý âm thanh chuẩn hóa

Với nhóm lightweight, audio được decode bằng SoundFile khi được hỗ trợ, lấy trung bình kênh để thành mono, resample bằng polyphase về 22.050 Hz và zero-pad/cắt thành 3,0 s (66.150 sample). Tín hiệu chia theo thời gian thành sáu segment không chồng lấp 0,5 s. Mỗi segment dùng STFT cửa sổ Hann, `n_fft=2048`, hop 512; Mel bank HTK tam giác 128 band từ 20–8.000 Hz. Power được đổi sang dB tương đối với cực đại segment, cắt trong 80 dB, tuyến tính hóa về [0,255], resize bilinear thành 224×224 và lặp thành ba kênh. Input cuối là float32 `6×224×224×3`.

Với segment $t$, $X_t(m,k)=\sum_{n=0}^{N-1}x_t[n+mH]w[n]e^{-j2\pi kn/N}$, trong đó $N=2048$, $H=512$. Mel power là $S_{mel,t}(m,r)=\sum_k|X_t(m,k)|^2H_r(k)$; code floor ở $10^{-10}$, lấy $10\log_{10}$, trừ cực đại segment và clip $[-80,0]$ dB. Hình 4 nhấn mạnh thứ tự thời gian được giữ nguyên.

![Hình 4. Tiền xử lý chuẩn hóa và temporal classifier của bốn detector lightweight.](figures/lightweight_temporal_pipeline.png)

Adapter tham chiếu dùng waveform mono 16 kHz dài 64.600 sample (4,0375 s), polyphase resampling và prefix/zero-padding. Loader gốc dùng librosa và repetition padding cho clip ngắn. Sai khác đã được ghi nhận và không bị thay đổi sau khi xem test. Parity native–ONNX chỉ chứng minh export trên cùng tensor adapter.

### 3.4 Các detector

**Bảng 4. Cấu hình detector và provenance.**

| Detector | Nhóm | Runtime | Input | Cơ chế | Tham số | Provenance |
|---|---|---|---|---|---:|---|
| MobileNetV3Small-LSTM | lightweight | TensorFlow 2.15 | Mel, 3,0 s | MobileNet 576-D + LSTM(128) | 1.308.401 | LAVA; ImageNet |
| EfficientNet-B0-LSTM | lightweight | TensorFlow 2.15 | Mel, 3,0 s | EfficientNet 1280-D + LSTM(128) | 4.779.300 | LAVA; ImageNet; warm-up best |
| MnasNet-A1-LSTM | lightweight | TensorFlow 2.15 | Mel, 3,0 s | MnasNet 1280-D + LSTM(128) | 3.369.255 | LAVA; scratch; early-stopped best |
| ShuffleNetV2-1.0x-LSTM | lightweight | TensorFlow 2.15 | Mel, 3,0 s | ShuffleNetV2 1024-D + LSTM(128) | 1.868.441 | LAVA; scratch; early-stopped best |
| RawNet2 | tham chiếu | ONNX từ PyTorch | waveform, 4,0375 s | Sinc/residual/attention/GRU | 17.621.410 | checkpoint ngoài; README: ASVspoof 2019 LA |
| AASIST | tham chiếu | ONNX từ PyTorch | waveform, 4,0375 s | Sinc/encoder/heterogeneous graph attention | 297.866 | checkpoint ngoài; README: ASVspoof 2019 LA |

Nhóm lightweight dùng chung `TimeDistributed(backbone) → LSTM(128) → Dense(64, ReLU) → Dropout(0,4) → sigmoid`; không ép embedding về cùng chiều. RawNet2/AASIST giữ kiến trúc native. Keras count gồm state BatchNorm không trainable; native PyTorch count không gồm buffers.

Với chuỗi $\mathbf{X}=\{X_1,\ldots,X_6\}$, backbone dùng chung sinh $\mathbf{z}_t=f_\theta(X_t)$, LSTM sinh $\mathbf{h}=\operatorname{LSTM}(\mathbf{z}_1,\ldots,\mathbf{z}_6)$ và head trả $p_{fake}=\sigma(W_2\operatorname{Dropout}_{0,4}(\operatorname{ReLU}(W_1\mathbf{h}+b_1))+b_2)$. MobileNet dùng inverted residual/SE và embedding 576-D; ShuffleNet dùng channel split, pointwise/depthwise convolution, concatenate và channel shuffle với embedding 1024-D; MnasNet và EfficientNet xuất embedding 1280-D.

RawNet2 nhận 64.600 waveform sample, dùng front end kiểu Sinc, residual block, channel attention, GRU và classifier [9,26]. AASIST dùng waveform front end, encoder 2-D, graph phổ và thời gian, heterogeneous graph attention, master node, pooling và readout [10,27]. Adapter chỉ ánh xạ spoof posterior về $P(FAKE)$, không thay kiến trúc.

### 3.5 Huấn luyện và provenance artifact

MobileNet dùng ImageNet và lifecycle hai stage: warm-up head khi backbone đóng băng, sau đó partial fine-tuning LR thấp với BatchNorm đóng băng. EfficientNet cũng khởi tạo ImageNet nhưng deployment hiện tại chỉ là checkpoint warm-up tốt nhất ở epoch 47 (`val_loss=0,1698`), chưa phải lifecycle fine-tune/global-best hoàn chỉnh. MnasNet được train scratch end-to-end từ epoch 1; artifact là best epoch 27 của run early stop ở epoch 39, với Adam `1e-4`, clip norm 1,0, label smoothing 0,1 và L2 `1e-5` theo implementation production.

RawNet2/AASIST không được LAVA train. Checkpoint `.pth` được strict-load rồi export ONNX. Sáu tensor audio thật cho sai khác `P(FAKE)` tối đa lần lượt `4,11×10⁻⁶` và `5,96×10⁻⁸`; đây là parity export, không phải bằng chứng cùng training pipeline.

ShuffleNet dùng Adam LR đầu $3\times10^{-4}$, 56/56 BatchNorm trainable, dừng ở epoch 28 và restore epoch 16; threshold validation là 0,12. MnasNet dùng Adam $10^{-4}$, clipnorm 1,0, label smoothing 0,1, weight decay $10^{-5}$, 49/49 BatchNorm trainable và restore epoch 27. Hình 5 và Bảng 5 tách rõ provenance thay vì mô tả sáu model như được train đồng nhất.

![Hình 5. Chiến lược khởi tạo, huấn luyện và nguồn checkpoint.](figures/training_provenance_strategies.png)

**Bảng 5. Chính sách huấn luyện, lựa chọn và threshold.**

| Detector | Khởi tạo / chế độ | LR đầu | BN | Artifact chọn | Threshold |
|---|---|---:|---|---|---|
| MobileNetV3 | ImageNet; warm-up + fine-tune | $10^{-4}$ | đóng băng khi fine-tune | lifecycle cục bộ | validation F1, 0,82 |
| EfficientNet-B0 | ImageNet; warm-up | $10^{-4}$ | backbone đóng băng | epoch 47 warm-up | validation F1, 0,90 |
| MnasNet-A1 | scratch end-to-end | $10^{-4}$ | 49/49 trainable | epoch 27/39 | validation F1, 0,90 |
| ShuffleNetV2 | scratch end-to-end | $3\times10^{-4}$ | 56/56 trainable | epoch 16/28 | validation F1, 0,12 |
| RawNet2 | checkpoint ngoài | N/A | native | strict-load + ONNX | mặc định 0,50 |
| AASIST | checkpoint ngoài | N/A | native | strict-load + ONNX | mặc định 0,50 |

### 3.6 Semantics điểm và threshold

LAVA cố định REAL=0, FAKE=1; mọi adapter trả (p=P(FAKE)):

$$\hat y=\begin{cases}1,&p\ge\tau,\\0,&p<\tau.\end{cases}\tag{1}$$

Threshold lightweight là 0,82 (MobileNet) và 0,90 (EfficientNet/MnasNet), được hiệu chỉnh theo F1 lớp FAKE trên validation. Hai reference giữ threshold mặc định 0,5 chưa calibrate. Test không chọn weight hay threshold, và score không được tuyên bố là xác suất ngoài đời đã calibration.

### 3.7 Giao thức clean

Toàn bộ 2.737 test sample (1.575 REAL, 1.162 FAKE) được dùng. Với FAKE là positive:

$$Precision=\frac{TP}{TP+FP},\quad Recall=\frac{TP}{TP+FN},\quad F1=\frac{2PR}{P+R}.\tag{2}$$

Ngoài ra, $Accuracy=(TP+TN)/(TP+TN+FP+FN)$ và $F1_{macro}=\frac{1}{2}(F1_{REAL}+F1_{FAKE})$. Sigmoid classifier cục bộ dùng binary cross-entropy $\mathcal{L}_{BCE}=-N^{-1}\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]$; riêng optimization profile MnasNet có label smoothing/weight decay đã ghi trong metadata, không được suy rộng cho model khác.

ROC-AUC dùng raw (P(FAKE)). Với threshold (\tau):

$$FAR(\tau)=\frac{FP}{FP+TN},\qquad FRR(\tau)=\frac{FN}{FN+TP}.\tag{3}$$

Implementation nội suy tại điểm đổi dấu đầu tiên của (FAR-FRR):

$$EER=\frac{FAR(\tau^*)+FRR(\tau^*)}{2},\quad FAR(\tau^*)\simeq FRR(\tau^*).\tag{4}$$

### 3.8 Stress test robustness

Do ước lượng runtime, robustness toàn test chưa chạy. Trước khi xem lỗi model, một subset phân tầng deterministic 100 sample (58 REAL, 42 FAKE; seed 42) được chọn. Clean baseline lấy đúng các ID này; output và figure được tách riêng, đánh dấu diagnostic.

Noise là AWGN có seed ở 20/10/5/0 dB theo RMS toàn prefix, lưu WAV float32. Compression dùng FFmpeg: MP3 128/64 kb/s, Opus 64 kb/s, AAC 96 kb/s; PCM16 là một phần round trip. Replay mô phỏng dùng tap 0/17/43/89 ms, gain 1/0,45/0,25/0,12, chuẩn hóa cố định và Butterworth band-pass bậc bốn 100–3.800 Hz. Đây không phải physical replay hay measured RIR. (\Delta F1=F1_{clean}-F1_{stress}), thấp hơn tốt hơn; mean lấy đều chín condition, không dùng trọng số tùy ý. Không có unseen dataset result.

Với $P_s$ là công suất tín hiệu và SNR mục tiêu $r$, code đặt $P_n=P_s/10^{r/10}$. Seed riêng của sample lấy từ 16 ký tự hex đầu của SHA-256. Tương tự, $\Delta AUC_c=AUC_{clean}-AUC_c$, $\Delta EER_c=EER_c-EER_{clean}$ và mean F1 degradation là $K^{-1}\sum_{k=1}^{K}\Delta F1_k$, $K=9$. Encoder version của FFmpeg không được sealed nên được đánh dấu `NOT_MEASURED`.

### 3.9 Hiệu quả tính toán

Mỗi model chạy tuần tự trong tiến trình cô lập trên cùng CPU Windows desktop, float32, batch 1, một computational thread. Metadata sealed ghi Windows `10.0.26200` và `AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD`; tên thương mại CPU và dung lượng RAM cài đặt không được lưu nên paper không tự suy đoán. Không dùng accelerator. Lightweight chạy TensorFlow 2.15; reference export chạy ONNX Runtime 1.17.3. Mười warm-up đi trước 50 lần đo. Hệ thống lưu mean/median/std/P95, preprocessing, model-only, end-to-end, throughput, load và RSS tiến trình lấy mẫu mỗi 20 ms. Startup và graph tracing bị loại; RSS không phải model-only allocation.

$$\bar T=\frac{1}{N}\sum_iT_i,\quad \sigma_T=\sqrt{\frac{1}{N-1}\sum_i(T_i-\bar T)^2},$$

$$RTF=\frac{T_{processing}}{T_{audio}},\qquad Throughput=\frac{N_{clips}}{T_{total}},\qquad S_{model}=\frac{bytes}{2^{20}}.\tag{5}$$

RTF<1 chỉ nghĩa là offline processing nhanh hơn thời lượng clip, không chứng minh causal streaming hay edge deployment.

### 3.10 Pareto

Pareto diagnostic tối thiểu hóa EER clean-subset, mean ΔF1 và RTF end-to-end. (A) trội (B) khi:

$$f_i(A)\le f_i(B)\ \forall i,\qquad \exists j:f_j(A)<f_j(B).\tag{6}$$

Không có weighted score. Vì robustness chỉ là subset, frontier chính thức vẫn `NOT_RUN`.

### 3.11 Thống kê và lỗi

Scores clean hỗ trợ 1.000 bootstrap phân tầng dùng chung seed 42, tạo percentile CI 95% cho F1/AUC/EER. Đây là bất định test-sample, không phải multi-seed. Exact McNemar với Holm adjustment và paired bootstrap được áp dụng sau khi kiểm tra alignment ID; agreement/error overlap dùng cùng thứ tự sample.

Mọi score file được join theo `sample_id`. Bootstrap resample riêng chỉ số REAL và FAKE để giữ phân tầng. McNemar dùng hai ô bất đồng $n_{01}$ và $n_{10}$; Holm kiểm soát family-wise error trên đủ 15 cặp. Pipeline publication chỉ đọc CSV/JSON đã lưu, không import detector loader và không thể gọi train/inference.

## 4. Kết quả

### 4.1 Phạm vi artifact

Sáu artifact đều PASS load, kiểm tra score hữu hạn, probe hai lớp, hash và public-adapter parity. ShuffleNet khớp manifest chính tắc, có 1.868.441 tham số và PASS conversion parity.

### 4.2 RQ1—hiệu năng clean

**Bảng 6. Hiệu năng trên toàn test chính tắc.**

| Model | Accuracy | Precision | Recall | F1 | Macro F1 | AUC | EER | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MobileNetV3 | 0,9766 | 0,9741 | 0,9707 | **0,9724** | **0,9761** | **0,9911** | **0,0250** | 1545/30/34/1128 |
| EfficientNet-B0 | 0,9635 | 0,9724 | 0,9406 | 0,9563 | 0,9624 | 0,9877 | 0,0400 | 1544/31/69/1093 |
| MnasNet-A1 | 0,9697 | 0,9599 | 0,9690 | 0,9645 | 0,9690 | 0,9886 | 0,0310 | 1528/47/36/1126 |
| ShuffleNetV2 | **0,9850** | **0,9795** | **0,9854** | **0,9824** | **0,9847** | **0,9929** | **0,0146** | 1551/24/17/1145 |
| RawNet2 ngoài | 0,4936 | 0,4380 | 0,6807 | 0,5330 | 0,4900 | 0,5178 | 0,4813 | 560/1015/371/791 |
| AASIST ngoài | 0,5513 | 0,4758 | 0,5585 | 0,5139 | 0,5487 | 0,5597 | 0,4463 | 860/715/513/649 |

Bốn artifact lightweight vượt hai reference trong giao thức cụ thể này; ShuffleNet đứng đầu F1 0,9824, macro-F1 0,9847, AUC 0,9929 và EER 0,0146. MobileNet đứng thứ hai theo F1/AUC và nhanh nhất. Đây không phải ranking thuần kiến trúc vì initialization, provenance, threshold, duration và preprocessing khác nhau.

![Hình 6. Accuracy, F1, macro-F1 và ROC-AUC clean của sáu detector.](figures/clean_metric_grouped_6_models.png)

### 4.3 Đặc điểm lỗi theo lớp

**Bảng 7. Precision, recall và F1 theo lớp trên full test.**

| Model | REAL P | REAL R | REAL F1 | FAKE P | FAKE R | FAKE F1 |
|---|---:|---:|---:|---:|---:|---:|
| MobileNetV3 | 0,9785 | 0,9810 | 0,9797 | 0,9741 | 0,9707 | 0,9724 |
| EfficientNet-B0 | 0,9572 | 0,9803 | 0,9686 | 0,9724 | 0,9406 | 0,9563 |
| MnasNet-A1 | 0,9770 | 0,9702 | 0,9736 | 0,9599 | 0,9690 | 0,9645 |
| ShuffleNetV2 | 0,9892 | 0,9848 | 0,9870 | 0,9795 | 0,9854 | 0,9824 |
| RawNet2 | 0,6015 | 0,3556 | 0,4469 | 0,4380 | 0,6807 | 0,5330 |
| AASIST | 0,6264 | 0,5460 | 0,5834 | 0,4758 | 0,5585 | 0,5139 |

ShuffleNet có 24 FP và 17 FN, ít nhất trong sáu model. EfficientNet mất recall FAKE với 69 FN; MnasNet đổi lại 47 FP để giảm FN xuống 36. RawNet2 tạo 1.015 FP, phù hợp với khả năng dịch domain/input và threshold ngoài chưa hiệu chỉnh.

![Hình 7. Confusion matrix full-test của sáu detector.](figures/confusion_matrix_panel_6_models.png)

### 4.4 Phân tích ROC và DET

ROC và DET dùng raw $P(FAKE)$ nên độc lập với operating threshold. Bốn lightweight có separation mạnh, trong khi hai reference gần đường ngẫu nhiên hơn trên dataset và adapter hiện tại.

![Hình 8. ROC trên toàn bộ 2.737 test sample.](figures/roc_comparison_6_models.png)

![Hình 9. DET và EER trên clean test.](figures/det_comparison_6_models.png)

### 4.5 RQ2—robustness noise diagnostic

**Bảng 8. Mean ΔF1 trên 100 sample ghép cặp; thấp hơn tốt hơn.**

| Model | Clean F1 | Noise | Codec | Replay mô phỏng | Mean 9 condition |
|---|---:|---:|---:|---:|---:|
| MobileNetV3 | 0,9756 | 0,6722 | 0,0095 | 0,0620 | 0,3099 |
| EfficientNet-B0 | 0,9630 | 0,8053 | 0,0000 | 0,0641 | 0,3650 |
| MnasNet-A1 | 0,9756 | 0,6119 | 0,0233 | 0,1489 | 0,2989 |
| ShuffleNetV2 | 0,9756 | 0,3555 | 0,0032 | 0,0256 | 0,1623 |
| RawNet2 ngoài | 0,5660 | 0,5077 | −0,0007 | −0,0109 | 0,2241 |
| AASIST ngoài | 0,5060 | −0,0660 | −0,0074 | −0,0682 | −0,0402 |

Codec chỉ làm thay đổi F1 nhỏ ở lightweight, trong khi AWGN gây suy giảm lớn. Replay đưa F1 về 0,9136/0,8989/0,8267 tương ứng MobileNet/EfficientNet/MnasNet. Suy giảm âm của baseline ngoài yếu không chứng minh robustness tốt; subset nhỏ có thể đổi score distribution ngẫu nhiên.

![Hình 10. F1 diagnostic theo mức AWGN.](figures/noise_f1_vs_snr_6_models.png)

### 4.6 Robustness compression và replay mô phỏng

Bốn codec round trip chỉ làm mean F1 của lightweight thay đổi từ 0,0000 đến 0,0233, nhỏ hơn nhiều so với AWGN. ShuffleNet đạt replay F1 0,9500, giảm 0,0256; MobileNet/EfficientNet giảm khoảng 0,062–0,064 và MnasNet giảm 0,1489. Delta âm của reference bắt đầu từ baseline thấp, không chứng minh distortion cải thiện detector.

![Hình 11. F1 dưới bốn condition codec diagnostic.](figures/codec_f1_comparison_6_models.png)

![Hình 12. So sánh clean-subset và simulated replay.](figures/replay_f1_comparison_6_models.png)

### 4.7 Tổng hợp robustness

Trong chín condition, ShuffleNet có mean degradation dương thấp nhất trong bốn lightweight (0,1623), tiếp theo MnasNet 0,2989, MobileNet 0,3099 và EfficientNet 0,3650. AASIST có delta âm nhưng clean-subset F1 chỉ 0,5060; vì vậy phải đọc absolute metric cùng degradation.

![Hình 13. Heatmap ΔF1 theo condition.](figures/robustness_heatmap_6_models.png)

### 4.8 Hiệu quả tính toán

**Bảng 9. Hiệu quả CPU desktop một luồng.**

| Model | Params | MiB | RSS MiB | Pre ms | Model ms | Mean/P95 ms | Clips/s | RTF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV3 | 1.308.401 | 5,64 | 554,6 | 13,43 | 30,07 | **43,81/52,12** | **22,83** | **0,0146** |
| EfficientNet-B0 | 4.779.300 | 18,96 | 651,2 | 13,59 | 168,36 | 175,85/196,11 | 5,69 | 0,0586 |
| MnasNet-A1 | 3.369.255 | 13,43 | 566,0 | 13,60 | 95,25 | 117,00/138,96 | 8,55 | 0,0390 |
| ShuffleNetV2 | 1.868.441 | 7,74 | 501,8 | 15,57 | 48,05 | 62,52/75,37 | 15,99 | 0,0208 |
| RawNet2 ngoài | 17.621.410 | 67,65 | 249,5 | 0,63 | 95,10 | 96,54/102,62 | 10,36 | 0,0239 |
| AASIST ngoài | 297.866 | **1,61** | 442,5 | **0,47** | 359,61 | 362,91/398,45 | 2,76 | 0,0899 |

MobileNet có latency/RTF tốt nhất; AASIST có file nhỏ nhất nhưng latency lớn nhất. Param hay size riêng lẻ không dự báo runtime. Mọi RTF<0,1 nhưng không phải streaming/edge validation.

![Hình 14. Độ trễ end-to-end trong giao thức CPU chung.](figures/end_to_end_latency_bar_6_models.png)

![Hình 15. Số tham số theo quy ước count đã nêu.](figures/parameters_bar_6_models.png)

MobileNet đạt 22,83 clip/s và RTF 0,0146; ShuffleNet đạt 15,99 clip/s và RTF 0,0208. AASIST có artifact nhỏ nhất nhưng model-only latency lớn nhất, cho thấy parameter count không dự báo trực tiếp runtime. RSS là toàn tiến trình, biến thiên từ 249,5 đến 651,2 MiB. Tất cả RTF dưới 0,1 chỉ chứng minh inference offline nhanh hơn thời lượng âm thanh trên CPU đã đo.

### 4.9 RQ3—Pareto thăm dò

**Bảng 10. Pareto diagnostic ba mục tiêu.**

| Model | EER subset | Mean ΔF1 | RTF | Pareto? |
|---|---:|---:|---:|---|
| MobileNetV3 | 0,0476 | 0,3099 | 0,0146 | Có |
| EfficientNet-B0 | 0,0476 | 0,3650 | 0,0586 | Không |
| MnasNet-A1 | 0,0476 | 0,2989 | 0,0390 | Không |
| ShuffleNetV2 | 0,0476 | 0,1623 | 0,0208 | Có |
| RawNet2 ngoài | 0,4138 | 0,2241 | 0,0239 | Không |
| AASIST ngoài | 0,3810 | −0,0402 | 0,0899 | Có |

Frontier chẩn đoán gồm MobileNet, ShuffleNet và AASIST. ShuffleNet trội MnasNet và RawNet2; MobileNet trội EfficientNet. AASIST còn trên frontier một phần vì baseline clean yếu tạo degradation âm, nên phải đọc cùng chất lượng tuyệt đối.

![Hình 16. Phép chiếu EER–RTF của Pareto diagnostic.](figures/pareto_eer_rtf_6_models.png)

Hình 17 cho thấy projection 2-D không biểu diễn hết quan hệ: MobileNet là cực trị tốc độ; ShuffleNet giữ cùng diagnostic EER nhưng degradation thấp hơn với RTF tăng vừa phải; AASIST không bị trội do tọa độ degradation âm xuất phát từ baseline yếu. Không có một “winner” duy nhất.

![Hình 17. Không gian Pareto diagnostic ba mục tiêu.](figures/pareto_3d_6_models.png)

### 4.10 Phân tích lỗi và agreement

Trên full clean test, 911 sample được cả sáu model dự đoán đúng và 10 sample bị cả sáu dự đoán sai. Bốn lightweight đồng thuận đúng trong khi cả hai reference sai ở 785 sample; chiều ngược lại có ba sample. ShuffleNet agreement với MobileNet/EfficientNet/MnasNet lần lượt là 0,979/0,970/0,977.

![Hình 18. Agreement theo cặp trên cùng full-test ID.](figures/agreement_heatmap_6_models.png)

### 4.11 Bất định thống kê

**Bảng 11. Khoảng tin cậy percentile 95% trên toàn bộ 2.737 mẫu test, từ 1.000 bootstrap phân tầng cùng phạm vi.**

| Model | F1 CI | AUC CI | EER CI |
|---|---|---|---|
| MobileNetV3 | [0,9654; 0,9788] | [0,9871; 0,9947] | [0,0178; 0,0311] |
| EfficientNet | [0,9477; 0,9651] | [0,9833; 0,9917] | [0,0311; 0,0482] |
| MnasNet | [0,9573; 0,9717] | [0,9842; 0,9923] | [0,0250; 0,0379] |
| RawNet2 | [0,5150; 0,5489] | [0,4944; 0,5395] | [0,4616; 0,5016] |
| AASIST | [0,4928; 0,5349] | [0,5377; 0,5815] | [0,4279; 0,4660] |
| ShuffleNetV2 | [0,9768; 0,9875] | [0,9894; 0,9960] | [0,0103; 0,0207] |

Tất cả sáu hàng hiện dùng cùng phạm vi full clean 2.737 mẫu; không còn trộn CI diagnostic 100 mẫu vào bảng này. Với họ 15 cặp McNemar exact, ShuffleNet khác MobileNet sau hiệu chỉnh Holm ($p_{adj}=0,0096$), và khác EfficientNet/MnasNet mạnh hơn. MobileNet–MnasNet ($p_{adj}=0,1060$) và EfficientNet–MnasNet ($p_{adj}=0,1180$) không đạt mức 0,05. Toàn bộ CI sai khác F1 và p-value nằm trong `table_12_pairwise_full_test.csv`. Ý nghĩa thống kê trên test này không loại bỏ nhiễu do domain và provenance.

**Bảng 12. So sánh ghép cặp full-test liên quan ShuffleNet. CI F1 là ShuffleNet trừ model đối chiếu; p-value đã hiệu chỉnh Holm trên 15 cặp.**

| Model đối chiếu | Discordant (Shuffle sai/đúng) | 95% CI chênh F1 | Adjusted p |
|---|---:|---:|---:|
| MobileNetV3 | 17 / 40 | [0,0036; 0,0166] | 0,0096 |
| EfficientNet-B0 | 12 / 71 | [0,0182; 0,0339] | $1,68\times10^{-10}$ |
| MnasNet-A1 | 10 / 52 | [0,0116; 0,0247] | $3,43\times10^{-7}$ |
| RawNet2 | 20 / 1.365 | [0,4322; 0,4677] | $<10^{-300}$ |
| AASIST | 10 / 1.197 | [0,4476; 0,4902] | $<10^{-300}$ |

### 4.12 Thảo luận và trả lời trực tiếp RQ

RQ1 cho thấy bốn hệ thống lightweight cục bộ vượt reference ngoài trên clean protocol; ShuffleNet có detection aggregate tốt nhất và MobileNet có runtime thấp nhất. RQ2 cho thấy codec ít gây hại hơn AWGN. RQ3 cho thấy Pareto membership không phải xếp hạng; frontier có MobileNet, ShuffleNet và AASIST nhưng có thể đổi theo hardware hoặc full-test robustness.

## 5. Hạn chế

Clean evaluation đã có sáu detector nhưng robustness chỉ có 100 sample; full-test robustness là `NOT_RUN`. Provenance dị thể: MobileNet/EfficientNet dùng ImageNet, MnasNet/ShuffleNet scratch, EfficientNet chỉ warm-up, RawNet2/AASIST là checkpoint ngoài. Metadata speaker/source/generator thiếu nên chỉ claim checksum-group-disjoint. Noise là AWGN, replay chỉ mô phỏng, không có unseen dataset. Timing chỉ trên một CPU Windows desktop, RSS là toàn tiến trình. Mỗi artifact chỉ đại diện một run; bootstrap test không thay multi-seed.

## 6. Hướng nghiên cứu tiếp theo

Ưu tiên đầu tiên không phải một lần sweep kiến trúc in-domain khác mà là corpus độc lập có metadata speaker, generator và source. Metadata đó cho phép chia speaker-, generator- và source-disjoint và kiểm tra ranking lightweight khi đổi domain. Cần chạy đủ chín stress condition trên toàn test trước khi bổ sung perturbation mới, để thay tọa độ robustness/Pareto diagnostic bằng ước lượng trên toàn bộ tập test.

Physical replay cần được thu thật qua nhiều phòng, loa, microphone, khoảng cách và mức âm lượng. Đánh giá deployment cần chạy trên Raspberry Pi, Jetson, điện thoại hoặc phần cứng mục tiêu tương đương, đồng thời tách cold start, offline clip và causal streaming. Quantization, pruning, ONNX/TFLite và quản lý state streaming là hướng kỹ thuật tiềm năng, chưa phải kết quả hiện tại. Cuối cùng, huấn luyện nhiều seed cho model cục bộ và calibration validation cho reference sẽ tách biến thiên checkpoint khỏi bất định do test sample.

## 7. Kết luận

LAVA cho thấy cách so sánh sáu detector giọng nói dị thể mà không phá kiến trúc native. ShuffleNetV2-LSTM đạt F1, AUC và EER clean tốt nhất; MobileNetV3Small-LSTM có latency thấp nhất. Robustness diagnostic cho thấy codec ít gây suy giảm hơn AWGN SNR thấp. Cần full-test robustness, physical replay, unseen dataset, edge hardware và nhiều seed trước khi tuyên bố LAVA được xác minh đầy đủ.

## Lời cảm ơn

Phần mềm và artifact thực nghiệm được phát triển bởi Phan Khắc Anh Tuấn, Nguyễn Phương Chinh, Lại Thành Đạt, Nguyễn Tấn Khiêm và Trương Thành Đạt. Bài báo không tuyên bố nguồn tài trợ bên ngoài.

## Phụ lục A. Khả năng tái lập và artifact phần mềm

LAVA-5 lịch sử được giữ ở `outputs/lava_5/`. ShuffleNet-only measurements và aggregate sáu model nằm ở `outputs/lava_6/`. `benchmark/lava6_incremental.py` chỉ chạy phần thiếu của ShuffleNet; `benchmark/lava6_report.py` ghép số liệu đã lưu. Quy trình không gọi `train.py`.

## Phụ lục B. Contract detector và score

Sáu registry ID là `mobilenetv3_lstm`, `shufflenetv2_lstm`, `mnasnet_lstm`, `efficientnet_b0_lstm`, `rawnet2` và `aasist`. Mỗi bundle production có model artifact, `threshold.json` và `metadata.json`. Sigmoid score của lightweight và logits/posterior native của reference đều được adapter chuyển thành một cột `p_fake`, với REAL=0 và FAKE=1.

## Phụ lục C. Condition robustness

Manifest diagnostic gồm matched clean, AWGN 20/10/5/0 dB, MP3 128/64 kb/s, Opus 64 kb/s, AAC 96 kb/s và một simulated replay. Audio condition được sinh một lần và dùng lại. Không có physical replay hoặc unseen corpus. Per-condition metrics nằm trong `outputs/lava_6/robustness/`.

## Phụ lục D. Biên đo hiệu quả

Preprocessing latency tính từ decode file đến tensor sẵn sàng cho detector. Model-only bắt đầu sau tensor đó. End-to-end gồm cả hai trong tiến trình warm; load time đo riêng. Throughput là nghịch đảo mean end-to-end khi chạy tuần tự batch một; RTF chia latency cho duration do detector khai báo. Interpreter launch và framework import không nằm trong warm latency.

## Phụ lục E. Output thống kê

`papers/tables/table_11_bootstrap_ci.csv` chứa CI full-test đồng nhất cho cả sáu model. `table_12_pairwise_full_test.csv` chứa 15 cặp với discordance, exact p-value, Holm p-value và paired F1-difference CI. Chúng thay thế draft cũ bị trộn phạm vi; không model nào được inference lại để sửa thống kê.

## Phụ lục F. Truy vết publication

`PAPER_EVIDENCE_MAP.md` ánh xạ claim sang evidence. `FIGURE_MANIFEST.md` và `TABLE_MANIFEST.md` ghi nguồn sinh. Các audit numerical, figure, table, reference và consistency là gate biên tập cuối. Chúng không thay thế independent replication nhưng làm ranh giới bằng chứng có thể kiểm tra.

## Tài liệu tham khảo

[1] Wu, Zhizheng; Kinnunen, Tomi; Evans, Nicholas; Yamagishi, Junichi; Hanilci, Cemal; Sahidullah, Md.; Sizov, Aleksandr. “ASVspoof 2015: The First Automatic Speaker Verification Spoofing and Countermeasures Challenge.” *Interspeech*, 2015. doi: 10.21437/Interspeech.2015-462

[2] Kinnunen, Tomi; Sahidullah, Md.; Delgado, Hector; Todisco, Massimiliano; Evans, Nicholas; Yamagishi, Junichi; Lee, Kong Aik. “The ASVspoof 2017 Challenge: Assessing the Limits of Replay Spoofing Attack Detection.” *Interspeech*, 2017. doi: 10.21437/Interspeech.2017-1111

[3] Todisco, Massimiliano; Wang, Xin; Vestman, Ville; Sahidullah, Md.; Delgado, Hector; Nautsch, Andreas; Yamagishi, Junichi; Evans, Nicholas; Kinnunen, Tomi; Lee, Kong Aik. “ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection.” *Interspeech*, 2019. doi: 10.21437/Interspeech.2019-2249

[4] Yamagishi, Junichi; Wang, Xin; Todisco, Massimiliano; Sahidullah, Md.; Patino, Jose; Nautsch, Andreas; Liu, Xin; Lee, Kong Aik; Kinnunen, Tomi; Evans, Nicholas; Delgado, Hector. “ASVspoof 2021: Accelerating Progress in Spoofed and Deepfake Speech Detection.” *ASVspoof 2021 Workshop*, 2021. doi: 10.21437/ASVSPOOF.2021-8

[5] Kinnunen, Tomi; Lee, Kong Aik; Delgado, Hector; Evans, Nicholas; Todisco, Massimiliano; Sahidullah, Md.; Yamagishi, Junichi; Reynolds, Douglas A.. “t-DCF: A Detection Cost Function for the Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification.” *Odyssey*, 2018. doi: 10.21437/Odyssey.2018-44

[6] Frank, Joel; Sch"onherr, Lea. “WaveFake: A Data Set to Facilitate Audio Deepfake Detection.” *NeurIPS Datasets and Benchmarks*, 2021. https://arxiv.org/abs/2111.02813

[7] Yi, Jiangyan; Tao, Chenglong; Fu, Ruibo; Yan, Xinrui; Wang, Chenglong; Zhang, Tao; Zhang, Xiaohui; Zhao, Yan; Ren, Yong; Xu, Le; others. “Audio Deepfake Detection: A Survey.” *arXiv preprint arXiv:2308.14970*, 2023. https://arxiv.org/abs/2308.14970

[8] Li, Meng; Ahmadiadli, Yahang; Zhang, Xiao-Ping. “A Survey on Speech Deepfake Detection.” *arXiv preprint arXiv:2404.13914*, 2024. https://arxiv.org/abs/2404.13914

[9] Tak, Hemlata; Patino, Jose; Todisco, Massimiliano; Nautsch, Andreas; Evans, Nicholas; Larcher, Anthony. “End-to-End Anti-Spoofing with RawNet2.” *ICASSP*, 2021. doi: 10.1109/ICASSP39728.2021.9414234

[10] Jung, Jee-weon; Heo, Hee-Soo; Tak, Hemlata; Shim, Hye-jin; Chung, Joon Son; Lee, Bong-Jin; Yu, Ha-Jin; Evans, Nicholas. “AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks.” *ICASSP*, 2022. doi: 10.1109/ICASSP43922.2022.9747766

[11] Wu, Zhenzong; Das, Rohan Kumar; Yang, Jichen; Li, Haizhou. “Light Convolutional Neural Network with Feature Genuinization for Detection of Synthetic Speech Attacks.” *Interspeech*, 2020. doi: 10.21437/Interspeech.2020-1810

[12] Tak, Hemlata; Todisco, Massimiliano; Wang, Xin; Jung, Jee-weon; Yamagishi, Junichi; Evans, Nicholas. “Automatic Speaker Verification Spoofing and Deepfake Detection Using wav2vec 2.0 and Data Augmentation.” *Odyssey*, 2022. doi: 10.21437/Odyssey.2022-16

[13] Muller, Nicolas M.; Czempin, Pavel; Dieckmann, Franziska; Froghyar, Adam; Bottinger, Konstantin. “Does Audio Deepfake Detection Generalize?.” *Interspeech*, 2022. doi: 10.21437/Interspeech.2022-108

[14] Kawa, Piotr; Plata, Marcin; Syga, Piotr. “Attack Agnostic Dataset: Towards Generalization and Stabilization of Audio DeepFake Detection.” *Interspeech*, 2022. doi: 10.21437/Interspeech.2022-10078

[15] Muller, Nicolas M.; Evans, Nicholas; Tak, Hemlata; Sperl, Philip; Bottinger, Konstantin. “Harder or Different? Understanding Generalization of Audio Deepfake Detection.” *Interspeech*, 2024. doi: 10.21437/Interspeech.2024-247

[16] Pascu, Octavian; Stan, Adriana; Oneata, Dan; Oneata, Elisabeta; Cucu, Horia. “Towards Generalisable and Calibrated Audio Deepfake Detection with Self-Supervised Representations.” *Interspeech*, 2024. doi: 10.21437/Interspeech.2024-1302

[17] Li, Yuang; Zhang, Min; Ren, Mengxin; Qiao, Xiaosong; Ma, Miaomiao; Wei, Daimeng; Yang, Hao. “Cross-Domain Audio Deepfake Detection: Dataset and Analysis.” *EMNLP*, 2024. doi: 10.18653/v1/2024.emnlp-main.286

[18] Das, Arnab; El Kheir, Yassine; Franzreb, Carlos; Herzig, Tim; Polzehl, Tim; Moller, Sebastian. “Generalizable Audio Spoofing Detection Using Non-Semantic Representations.” *Interspeech*, 2025. doi: 10.21437/Interspeech.2025-1555

[19] Ciobanu, Ioan-Paul; Hiji, Andrei-Iulian; Ristea, Nicolae Catalin; Irofti, Paul; Rusu, Cristian; Ionescu, Radu Tudor. “XMAD-Bench: Cross-Domain Multilingual Audio Deepfake Benchmark.” *Findings of EACL*, 2026. doi: 10.18653/v1/2026.findings-eacl.162

[20] Howard, Andrew; Sandler, Mark; Chu, Grace; Chen, Liang-Chieh; Chen, Bo; Tan, Mingxing; Wang, Weijun; Zhu, Yukun; Pang, Ruoming; Vasudevan, Vijay; Le, Quoc V.; Adam, Hartwig. “Searching for MobileNetV3.” *ICCV*, 2019. doi: 10.1109/ICCV.2019.00140

[21] Sandler, Mark; Howard, Andrew; Zhu, Menglong; Zhmoginov, Andrey; Chen, Liang-Chieh. “MobileNetV2: Inverted Residuals and Linear Bottlenecks.” *CVPR*, 2018. doi: 10.1109/CVPR.2018.00474

[22] Zhang, Xiangyu; Zhou, Xinyu; Lin, Mengxiao; Sun, Jian. “ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices.” *CVPR*, 2018. doi: 10.1109/CVPR.2018.00716

[23] Ma, Ningning; Zhang, Xiangyu; Zheng, Hai-Tao; Sun, Jian. “ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design.” *ECCV*, 2018. doi: 10.1007/978-3-030-01264-9_8

[24] Tan, Mingxing; Le, Quoc V.. “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.” *ICML*, 2019. https://proceedings.mlr.press/v97/tan19a.html

[25] Tan, Mingxing; Chen, Bo; Pang, Ruoming; Vasudevan, Vijay; Sandler, Mark; Howard, Andrew; Le, Quoc V.. “MnasNet: Platform-Aware Neural Architecture Search for Mobile.” *CVPR*, 2019. doi: 10.1109/CVPR.2019.00293

[26] Ravanelli, Mirco; Bengio, Yoshua. “Speaker Recognition from Raw Waveform with SincNet.” *IEEE Spoken Language Technology Workshop*, 2018. doi: 10.1109/SLT.2018.8639585

[27] Velickovic, Petar; Cucurull, Guillem; Casanova, Arantxa; Romero, Adriana; Lio, Pietro; Bengio, Yoshua. “Graph Attention Networks.” *International Conference on Learning Representations*, 2018. https://openreview.net/forum?id=rJXMpikCZ

[28] Baevski, Alexei; Zhou, Yuhao; Mohamed, Abdelrahman; Auli, Michael. “wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.” *NeurIPS*, 2020. https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html

[29] Hsu, Wei-Ning; Bolte, Benjamin; Tsai, Yao-Hung Hubert; Lakhotia, Kushal; Salakhutdinov, Ruslan; Mohamed, Abdelrahman. “HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.” *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2021. doi: 10.1109/TASLP.2021.3122291

[30] Chen, Sanyuan; Wang, Chengyi; Chen, Zhengyang; Wu, Yu; Liu, Shujie; Chen, Zhuo; Li, Jinyu; Kanda, Naoyuki; Yoshioka, Takuya; Xiao, Xiong; others. “WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing.” *IEEE Journal of Selected Topics in Signal Processing*, 2022. doi: 10.1109/JSTSP.2022.3188113

[31] Efron, Bradley. “Bootstrap Methods: Another Look at the Jackknife.” *The Annals of Statistics*, 1979. doi: 10.1214/aos/1176344552

[32] McNemar, Quinn. “Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages.” *Psychometrika*, 1947. doi: 10.1007/BF02295996

[33] Holm, Sture. “A Simple Sequentially Rejective Multiple Test Procedure.” *Scandinavian Journal of Statistics*, 1979. https://www.jstor.org/stable/4615733

[34] McFee, Brian; Raffel, Colin; Liang, Dawen; Ellis, Daniel P. W.; McVicar, Matt; Battenberg, Eric; Nieto, Oriol. “librosa: Audio and Music Signal Analysis in Python.” *Proceedings of the 14th Python in Science Conference*, 2015. doi: 10.25080/Majora-7b98e3ed-003
