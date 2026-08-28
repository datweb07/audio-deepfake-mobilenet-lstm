# Audio Deepfake Detection - MobileNetV3Small + LSTM

Hệ thống phân loại nhị phân audio với quy ước thống nhất:

- `REAL = 0`
- `FAKE = 1`
- đầu ra sigmoid là `P(FAKE)`
- `P(FAKE) >= threshold` được phân loại là `FAKE`

Pipeline chính là: audio 3 giây -> 6 segment theo đúng thứ tự thời gian -> Mel spectrogram RGB 224x224 -> `TimeDistributed(MobileNetV3Small)` -> LSTM -> sigmoid.

## 1. Setup

Khuyến nghị dùng Python 3.11 và một virtual environment mới. Không cài requirements của ba repo reference.

```powershell
cd D:\audio-deepfake-mobilenet-lstm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Kiểm tra môi trường:

```powershell
python -m pip check
python -c "import tensorflow, librosa, cv2, sklearn, streamlit; print(tensorflow.__version__)"
```

TensorFlow tự dùng GPU tương thích nếu phát hiện được. Không có GPU thì project vẫn chạy bằng CPU, nhưng training chậm hơn đáng kể.

## 2. Dataset

Chỉ hai thư mục sau được scan làm dataset production:

```text
data\REAL\
data\FAKE\
```

Định dạng hỗ trợ: `.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`. Mỗi file được resample về 22,050 Hz, mono và lấy tối đa 3 giây đầu. File ngắn hơn được zero-pad.

Ba thư mục sau chỉ là nguồn tham khảo, không được scan làm dataset và không được import vào runtime:

```text
deepfake-audio-detection\
enhancing-deepfake-detection-using-mobilenet-lstm-hybrid-model-main\
mobilenetv3.pytorch\
```

Split được tạo lại một cách deterministic với seed 42 theo tỷ lệ 70% train, 15% validation, 15% test. Original files được split trước; augmentation chỉ chạy trên train.

## 3. Train

```powershell
python train.py
```

Training gồm:

1. Phase 1: đóng băng toàn bộ MobileNetV3Small, train LSTM và classifier với learning rate `1e-4`.
2. Phase 2: mở một phần cuối backbone, giữ toàn bộ BatchNormalization frozen, compile lại và fine-tune với learning rate `1e-5`.
3. Dùng validation probabilities để tìm threshold tối ưu theo F1 và lưu threshold.

Artifacts mới:

```text
models\best_model_phase1.keras
models\best_model_phase2.keras
models\best_threshold.txt
models\model_metadata.json
outputs\plots\training_history_phase1.png
outputs\plots\training_history_phase2.png
```

Các file `.h5` cũ được giữ lại để không làm mất dữ liệu, nhưng được xem là legacy. Chúng được train trước khi sửa đúng input scale của MobileNetV3, vì vậy phải retrain để có kết quả đáng tin cậy.

## 4. Evaluate

Chỉ chạy sau khi đã train lại:

```powershell
python evaluate.py
```

Script dùng test split độc lập và threshold đã tune trên validation. Output gồm Accuracy, Precision, Recall, F1, ROC-AUC từ raw `P(FAKE)`, confusion matrix và classification report.

## 5. Predict

```powershell
python predict.py --audio "data\REAL\biden-original.wav"
python predict.py --audio "data\FAKE\biden-to-linus.wav"
```

Output gồm prediction, confidence của class được chọn, raw `P(FAKE)` và threshold thực tế.

## 6. Streamlit

```powershell
python -m streamlit run app.py
```

Mở `http://localhost:8501`. UI hỗ trợ upload/playback, waveform, Mel spectrogram, REAL/FAKE, confidence, raw `P(FAKE)` và threshold. UI không hiển thị class prediction giả cho từng segment vì classifier chỉ dự đoán toàn sequence.

## 7. Retraining với dataset mới

1. Thay nội dung `data\REAL` và `data\FAKE`.
2. Chạy lại `python train.py`.
3. Các checkpoint `.keras`, metadata, plots và threshold sẽ được cập nhật.
4. Split được regenerate deterministic từ danh sách file mới.
5. Chạy `python evaluate.py`, sau đó smoke-test `predict.py` và Streamlit.

Không dùng threshold cũ với model mới và không dùng model cũ với preprocessing mới.

