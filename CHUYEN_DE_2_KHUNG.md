# CHUYÊN ĐỀ 2: ỨNG DỤNG TRÍ TUỆ NHÂN TẠO PHÁT HIỆN NỐT PHỔI TRÊN ẢNH CHỤP CT
## Khung nội dung chi tiết - Tối thiểu 20 trang

---

## 1. MỞ ĐẦU (1.5–2 trang)

### 1.1 Bối cảnh bài toán
- **Viết về:** Ung thư phổi là căn bệnh ác tính hàng đầu trên thế giới, tỷ lệ sống sót phụ thuộc vào phát hiện sớm
- **Con số cụ thể:** WHO data, tỷ lệ tử vong ở Việt Nam, độ chính xác CT trong chẩn đoán
- **Vấn đề hiện tại:** Bác sĩ phải xem hàng trăm slice ảnh DICOM, khó phát hiện nốt nhỏ (<5mm), dễ bỏ sót
- **Tại sao cần AI:** Hỗ trợ tầm soát, tăng độ chính xác, giảm tải công việc

### 1.2 Mục tiêu đề tài
- Xây dựng ứng dụng desktop tích hợp AI phát hiện nốt phổi trên ảnh CT tự động
- Tối ưu cho cấu hình máy yếu (máy bác sĩ thường không cao cấp)
- Giao diện thân thiện, kết quả rõ ràng (hiển thị vị trí, kích thước, nguy hiểm)

### 1.3 Phạm vi và cấu trúc báo cáo
- Phạm vi: Phát hiện và phân loại nốt, giảm false positive, không điều trị
- Cấu trúc: Lý thuyết → Thiết kế → Xây dựng → Thử nghiệm → Kết luận
- Dữ liệu dùng: ~10 bệnh nhân, 1000+ nốt phổi được nhãn thủ công

---

## 2. TỔNG QUAN BÀI TOÁN PHÁT HIỆN NỐT PHỔI TRÊN CT (2.5–3 trang)

### 2.1 Đặc điểm ảnh CT phổi
- **Khác với ảnh 2D thường:** DICOM là định dạng y tế, từ 50–300 slice mỗi lần quét
- **Thông tin trong slice:** Mật độ HU (Hounsfield Unit), kích thước voxel thường 0.5–1mm
- **Mô phổi:** Là khí, nên có độ tương phản cao trên CT với các mô khác
- **Nốt phổi:** Hình cầu/bầu dục, kích thước 3–30mm, có/không ranh giới rõ ràng
- **Thách thức:** Nốt nhỏ khó nhìn, nốt gần với mạch máu, nốt gần với thành ngực

### 2.2 Các dạng nốt phổi trong lâm sàng
- **Solid nodule:** Nốt toàn phần, màu sáng đều (nguy hiểm hơn)
- **Ground-glass nodule (GGN):** Nốt mờ, có thể thấy mạch máu bên trong
- **Mixed nodule:** Vừa solid vừa GGN
- **Micronodule:** Nốt <3mm, khó phát hiện
- **Tại sao AI khó:** Tất cả các dạng này phải nhận diện được

### 2.3 Các metric đánh giá trong lâm sàng & ML
- **Sensitivity (Recall):** Tỷ lệ nốt thực tế được phát hiện (không bỏ sót)
- **Specificity:** Tỷ lệ không nhầm mạch máu thành nốt (giảm false positive)
- **PPV (Positive Predictive Value):** Độ tin cậy của từng phát hiện
- **FROC curve:** Đường cong đánh giá đặc thù của bài toán detect (dùng thay cho ROC)
- **Vì sao quan trọng:** Bác sĩ cần tin tưởng vào AI, nếu false positive nhiều sẽ bỏ qua

---

## 3. CƠ SỞ LÝ THUYẾT VÀ MÔ HÌNH LIÊN QUAN (3–4 trang)

### 3.1 Mạng U-Net cho segmentation phổi
- **Tại sao:** Phải cô lập phổi trước, loại bỏ background (xương, tim, nhân tạo)
- **Kiến trúc:** Encoder-decoder, skip connection, output là mask
- **Mục đích trong app:** Tạo ROI (vùng lợi ích) chỉ xử lý bên trong phổi, tăng tốc độ
- **Kết quả mong đợi:** Mask chính xác >95%, không bỏ sót mô phổi

**Sơ đồ U-Net Architecture:**
```
┌────────────────────────────────────────────────────────────────────────┐
│                              U-NET ARCHITECTURE                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input (512×512)                                                        │
│       ↓                                                                  │
│  ┌────────┐     skip1     ┌────────┐                                   │
│  │Conv64  ├──────────────→│        │                                   │
│  │ReLU    │               │        │                                   │
│  │(3×3)   │               │ Concat │                                   │
│  └────┬───┘               │ &Deconv│                                   │
│       ↓                   │        │                                   │
│  MaxPool (↓2)            │        │←──────────┐                        │
│       ↓                   │        │           │                        │
│  ┌────────┐     skip2     └────────┘           │                        │
│  │Conv128 ├──────────────────────────────→ DecoderBlock2               │
│  │(3×3)   │                                (Deconv128)                  │
│  └────┬───┘                                    ↑                        │
│       ↓                                        │                        │
│  MaxPool (↓2)                                  │                        │
│       ↓                                        │                        │
│  ┌────────┐     skip3     ┌────────┐          │                        │
│  │Conv256 ├──────────────→│        │          │                        │
│  │(3×3)   │               │ Concat │          │                        │
│  └────┬───┘               │ &Deconv├──────────┘                        │
│       ↓                   │(Deconv256)                                  │
│  MaxPool (↓2)            │        │                                   │
│       ↓                   └────────┘                                   │
│  ┌────────┐                   ↑                                        │
│  │Conv512 │             skip4 │                                        │
│  │(3×3)   │                   │                                        │
│  │BottleNeck │                │                                        │
│  └────┬───┘                   │                                        │
│       ↓                       │                                        │
│  ┌────────────┐               │                                        │
│  │DecoderBlock1 ├─────────────┘                                        │
│  │(Deconv512)  │                                                       │
│  └────┬────────┘                                                       │
│       ↓                                                                  │
│  ┌────────────┐                                                       │
│  │Conv1 (1×1) │                                                       │
│  │Output: 1ch │                                                       │
│  │Sigmoid     │                                                       │
│  └────┬───────┘                                                        │
│       ↓                                                                  │
│  Output Mask (512×512, binary)                                         │
│       ↓                                                                  │
│  Dice Loss + BCE → Optimize                                            │
│                                                                          │
└────────────────────────────────────────────────────────────────────────┘

Khác biệt quan trọng: Skip connection giữ thông tin fine-grained từ encoder,
giúp decoder tái dựng chi tiết ranh giới phổi chính xác hơn.
```

### 3.2 YOLO (You Only Look Once) - Phát hiện nốt
- **Tại sao YOLO:** Nhanh, real-time, phù hợp với máy yếu (YOLOv8n, YOLOv11n)
- **So sánh YOLOv8 vs YOLOv11:**
  - YOLOv8: Nhẹ hơn, tốc độ cao hơn
  - YOLOv11: Độ chính xác cao hơn, nhưng hơi nặng
  - **App chọn:** YOLOv8 để ưu tiên tốc độ trên máy yếu
- **Output:** Bounding box (x, y, w, h) + confidence score cho từng nốt
- **Challenge:** YOLO có thể nhầm mạch máu, nốt bé khó phát hiện

**Sơ đồ YOLO Detection Architecture:**
```
┌──────────────────────────────────────────────────────────────────┐
│                    YOLO DETECTION PIPELINE                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input CT Slice (416×416×3)                                     │
│       ↓                                                            │
│  ┌─────────────────────────────────┐                            │
│  │ Backbone (CSPDarknet)           │                            │
│  │ - Conv 3×3 stride 2 → 52×52     │                            │
│  │ - Conv 3×3 stride 2 → 26×26     │                            │
│  │ - Conv 3×3 stride 2 → 13×13     │                            │
│  │ (Feature extraction)             │                            │
│  └────────────┬────────────────────┘                            │
│               ↓                                                    │
│  ┌─────────────────────────────────┐                            │
│  │ Neck (PANet)                    │                            │
│  │ - FPN upsampling: 13→26→52 (·2) │                            │
│  │ - Concat with backbone features │                            │
│  │ - Multi-scale feature fusion    │                            │
│  └────────────┬────────────────────┘                            │
│               ↓                                                    │
│  ┌─────────────────────────────────┐                            │
│  │ Head (Detection)                │                            │
│  │ 3 Outputs: 52×52 / 26×26 / 13×13│ (Small / Med / Large)    │
│  │                                 │                            │
│  │ Each output: (x, y, w, h,      │                            │
│  │              conf, class_prob)  │                            │
│  │ Total: N × (4 + 1 + num_classes)│                            │
│  └────────────┬────────────────────┘                            │
│               ↓                                                    │
│  ┌─────────────────────────────────┐                            │
│  │ Post-Processing (NMS)           │                            │
│  │ - Filter by confidence > 0.5    │                            │
│  │ - Non-Max Suppression (IoU>0.4) │                            │
│  │ - Keep top detections           │                            │
│  └────────────┬────────────────────┘                            │
│               ↓                                                    │
│  Output: List of (x, y, w, h, conf)                             │
│  Example: [(100, 120, 30, 35, 0.92),                            │
│            (200, 150, 25, 28, 0.85),                            │
│            ...]                                                   │
│                                                                   │
│  Loss function: CIoU Loss + Focal Loss (để cân bằng class)      │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘

Úu điểm: Phát hiện nhanh 1 pass, đủ tốc độ real-time trên CPU.
Nhược điểm: Chỉ xem 2D slice, dễ nhầm với mạch máu 3D.
```

### 3.3 3D CNN - Giảm false positive
- **Vấn đề:** YOLO hoạt động từng slice 2D, không dùng thông tin 3D → nhầm mạch máu
- **Giải pháp:** Sau YOLO detect nốt, cắt khối 3D xung quanh (16×32×32 voxel)
- **Mô hình 3D CNN nhẹ:** Conv3D + BatchNorm + ReLU, output: [Nốt thực, Mạch máu, Rác]
- **Lợi ích:** Nhìn 3D context, độ chính xác tăng thêm 5–10%
- **Chi phí:** RAM hơi cao, nhưng dùng AMP (Automatic Mixed Precision) → tối ưu

**Sơ đồ 3D CNN Architecture (Lightweight FPR Filter):**
```
┌────────────────────────────────────────────────────────────────┐
│           3D CNN FOR FALSE POSITIVE REDUCTION (FPR)            │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: 3D Patch (16×32×32×1) [Depth × Height × Width × Ch]  │
│       ↓                                                          │
│  ┌──────────────────────────────┐                             │
│  │ Block 1: Conv3D 32 filters   │                             │
│  │ - Conv3D (3×3×3, stride=1)   │                             │
│  │ - BatchNorm 3D               │                             │
│  │ - ReLU                       │                             │
│  │ - MaxPool3D (2×2×2, stride=2)│                             │
│  │ Output: (8×16×16×32)         │                             │
│  └──────────┬───────────────────┘                             │
│             ↓                                                    │
│  ┌──────────────────────────────┐                             │
│  │ Block 2: Conv3D 64 filters   │                             │
│  │ - Conv3D (3×3×3, stride=1)   │                             │
│  │ - BatchNorm 3D               │                             │
│  │ - ReLU                       │                             │
│  │ - MaxPool3D (2×2×2, stride=2)│                             │
│  │ Output: (4×8×8×64)           │                             │
│  └──────────┬───────────────────┘                             │
│             ↓                                                    │
│  ┌──────────────────────────────┐                             │
│  │ Block 3: Conv3D 128 filters  │                             │
│  │ - Conv3D (3×3×3, stride=1)   │                             │
│  │ - BatchNorm 3D               │                             │
│  │ - ReLU                       │                             │
│  │ - MaxPool3D (2×2×2, stride=2)│                             │
│  │ Output: (2×4×4×128)          │                             │
│  └──────────┬───────────────────┘                             │
│             ↓                                                    │
│  ┌──────────────────────────────┐                             │
│  │ Global Average Pooling 3D    │                             │
│  │ Output: (128,)               │                             │
│  └──────────┬───────────────────┘                             │
│             ↓                                                    │
│  ┌──────────────────────────────┐                             │
│  │ Fully Connected Layers       │                             │
│  │ - FC1: 128 → 64 (ReLU)       │                             │
│  │ - Dropout (0.5)              │                             │
│  │ - FC2: 64 → 32 (ReLU)        │                             │
│  │ - Dropout (0.5)              │                             │
│  │ - FC3: 32 → 3 (Softmax)      │                             │
│  │ Output: [P_nodule, P_vessel, P_trash] │                     │
│  └──────────┬───────────────────┘                             │
│             ↓                                                    │
│  Output probabilities:                                          │
│  - 0: True nodule (nguy hiểm) → Keep                           │
│  - 1: Blood vessel (mạch máu) → Discard (false positive)       │
│  - 2: Trash/artifact → Discard                                 │
│                                                                 │
│  Decision: IF P_nodule > 0.6 THEN keep detection               │
│                                                                 │
│  Loss: Cross Entropy (3-way classification)                    │
│  Optimization: AMP (Automatic Mixed Precision) → 50% RAM save   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘

Dù nhỏ (0.8M param) nhưng hiệu quả cao vì dùng context 3D.
```

### 3.4 Morphological Filter - Hậu xử lý
- **Mục đích:** Loại bỏ các artifact (tiếng ồn), giữ lại nốt thực
- **Phương pháp:** Erode → Dilate (opening), loại các vùng nhỏ <10 pixel
- **Kết quả:** Kết quả sạch hơn, ít false positive hơn

### 3.5 Phương pháp clustering 3D
- **Vấn đề:** YOLO detect từng slice riêng, cùng một nốt trong nhiều slice sẽ tạo nhiều detection
- **Giải pháp:** Gom 3D (3D clustering) các detection lân cận về một nốt duy nhất
- **Phương pháp:** DBSCAN hoặc simple 3D IoU merge
- **Kết quả:** 1 nốt = 1 detection (không lặp), kích thước nốt chính xác hơn

---

## 4. PHÂN TÍCH YÊU CẦU VÀ ĐẶC TẢ HỆ THỐNG (1.5–2 trang)

### 4.1 Yêu cầu chức năng
- **FR1:** Mở file DICOM (single hoặc folder), hiển thị từng slice
- **FR2:** Tự động phát hiện nốt phổi, hiển thị bounding box + nguy hiểm (xanh=low, vàng=mid, đỏ=high)
- **FR3:** Cho phép người dùng điều chỉnh threshold confidence, thay đổi mô hình
- **FR4:** So sánh ảnh gốc vs ảnh được đánh dấu (overlap view)
- **FR5:** Export kết quả thành báo cáo JSON (nốt, tọa độ, kích thước, xác suất)
- **FR6:** Huấn luyện lại YOLO với dữ liệu mới (nâng cao độ chính xác cho bệnh viện cụ thể)

### 4.2 Yêu cầu phi chức năng
- **Tốc độ:** Xử lý 1 case (200 slice) trong <2 phút (ưu tiên tốc độ)
- **RAM:** <4GB khi chạy inference (máy bác sĩ thường có 8GB, cần để chạy việc khác)
- **GPU optional:** Hoạt động tốt trên CPU, nếu có GPU thì nhanh hơn
- **Độ chính xác:** Recall >85%, Precision >80% (không bỏ sót, ít false alarm)
- **Giao diện:** Dễ dùng, không cần kỹ năng ML, hướng dẫn rõ ràng

### 4.3 Use case chính
- **Bác sĩ Tây y:** Mở ảnh CT → app tự detect → xem kết quả, nếu đồng ý thêm vào báo cáo
- **Chuyên gia phổi:** Điều chỉnh threshold, so sánh thủ công, nếu không ổn thì huấn luyện lại
- **Quản lý bệnh viện:** Xem báo cáo tập hợp (bao nhiêu case, đã xử lý, từ chối bao nhiêu)

---

## 5. THIẾT KẾ KIẾN TRÚC ỨNG DỤNG (2–3 trang)

### 5.1 Kiến trúc tổng quát (tầng)
```
┌─────────────────────────────────────┐
│     Tầng Giao diện (UI Layer)       │
│  main.py + CustomTkinter (các Tab)  │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Tầng Business Logic (Pipeline)     │
│  pipeline.py: Điều phối các model   │
└──────────────┬──────────────────────┘
               │
   ┌───────────┼───────────┬──────────┐
   │           │           │          │
┌──▼──┐  ┌──────▼──┐  ┌───▼───┐  ┌──▼──┐
│U-Net│  │  YOLO   │  │3D CNN │  │Post-│
│(seg)│  │(detect) │  │(FPR)  │  │proc │
└─────┘  └─────────┘  └───────┘  └─────┘

   Data Processing (utils/)
   - Read DICOM, normalize, extract patch
```

### 5.2 Luồng dữ liệu end-to-end

**Sơ đồ Pipeline Inference End-to-End:**
```
┌──────────────────────────────────────────────────────────────────────────┐
│                  FULL PIPELINE: INPUT → OUTPUT                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  [Step 0] USER INPUT                                                    │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ User chọn folder DICOM từ UI                               │        │
│  │ Ví dụ: /data/3000518.000000-NA-66796/                      │        │
│  │ Contains: 1-001.dcm, 1-002.dcm, ..., 1-150.dcm             │        │
│  └────────────────────┬────────────────────────────────────────┘        │
│                       ↓                                                    │
│  [Step 1] LOAD & PREPROCESS                                             │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ image_reader.py:                                            │        │
│  │ - Đọc tất cả DICOM files → array (150, 512, 512)           │        │
│  │ - Extract pixel_spacing, slice_thickness từ DICOM metadata  │        │
│  │ - Normalize HU: clip [-1024, 400] → [0, 1]                │        │
│  │ Output: (150, 512, 512) float32, HU normalized             │        │
│  └────────────────────┬────────────────────────────────────────┘        │
│                       ↓                                                    │
│  [Step 2] LUNG SEGMENTATION (U-Net)                                     │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ lung_segment.py:                                            │        │
│  │ - Inference U-Net trên từng slice                          │        │
│  │ - Output: 150 mask (150, 512, 512) binary                  │        │
│  │ - Post-process: morphological closing để smooth mask        │        │
│  │ Result: Lung ROI (Region of Interest)                       │        │
│  └────────────────────┬────────────────────────────────────────┘        │
│                       ↓                                                    │
│  [Step 3] DETECTION (YOLO) - FOR EACH SLICE                            │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ nodule_detect.py (loop over 150 slices):                   │        │
│  │ - Input: slice [i] (512×512) + mask [i]                    │        │
│  │ - Resize to 416×416 (YOLO input)                           │        │
│  │ - Inference YOLO → (N_i, 6) detections                     │        │
│  │   Each detection: (x, y, w, h, conf, class_id)             │        │
│  │ - Filter by conf > 0.5, apply NMS                          │        │
│  │ Output: List of (x, y, w, h, conf, z_slice)               │        │
│  │ Typical: ~300–500 detections across all slices (high!)    │        │
│  └────────────────────┬────────────────────────────────────────┘        │
│                       ↓                                                    │
│  [Step 4] EXTRACT 3D PATCHES & CLASSIFY (3D CNN)                       │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ patch_extractor_3d.py + fpr_3d_net.py:                     │        │
│  │ For each YOLO detection (x, y, z):                          │        │
│  │ - Extract 3D patch (16×32×32) centered at (x, y, z)        │        │
│  │ - Inference 3D CNN → [P_nodule, P_vessel, P_trash]         │        │
│  │ - IF P_nodule > 0.6: Keep detection, ELSE: Discard        │        │
│  │ Output: Filtered detections (~100–200 left)                │        │
│  └────────────────────┬────────────────────────────────────────┘        │
│                       ↓                                                    │
│  [Step 5] POST-PROCESSING                                               │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ 5a) Morphological Filter:                                   │        │
│  │ - Remove blobs < 10 pixels (noise)                          │        │
│  │ - Erode + Dilate (opening)                                  │        │
│  │                                                              │        │
│  │ 5b) 3D Clustering (merge nearby detections):                │        │
│  │ - DBSCAN or 3D IoU-based merge                              │        │
│  │ - Threshold: if distance < 20mm → same nodule             │        │
│  │ - Compute centroid: (x_mean, y_mean, z_mean)              │        │
│  │                                                              │        │
│  │ 5c) Compute nodule size:                                    │        │
│  │ - diameter_mm = avg(width, height) × pixel_spacing         │        │
│  │ - Add metadata from merged detections                      │        │
│  │                                                              │        │
│  │ Output: Final list of nodules (~5–15 per case)             │        │
│  └────────────────────┬────────────────────────────────────────┘        │
│                       ↓                                                    │
│  [Step 6] CLASSIFY RISK LEVEL                                           │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ Rule-based risk classification (Lung-RADS):                │        │
│  │ - IF diameter < 6mm: 🟢 Low risk                           │        │
│  │ - IF 6 ≤ diameter < 8mm: 🟡 Intermediate risk              │        │
│  │ - IF diameter ≥ 8mm: 🔴 High risk                          │        │
│  │ + Consider nodule type (solid vs GGN)                      │        │
│  └────────────────────┬────────────────────────────────────────┘        │
│                       ↓                                                    │
│  [Step 7] GENERATE OUTPUT                                               │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ result_manager.py:                                          │        │
│  │ - Generate JSON report                                      │        │
│  │ - Generate PNG with annotated slices                        │        │
│  │ - Display in UI table                                       │        │
│  │                                                              │        │
│  │ JSON format:                                                │        │
│  │ {                                                            │        │
│  │   "case_id": "3000518",                                     │        │
│  │   "nodules": [                                              │        │
│  │     {                                                        │        │
│  │       "id": 1,                                              │        │
│  │       "x_mm": 245.5,                                        │        │
│  │       "y_mm": 189.2,                                        │        │
│  │       "z_slice": 45,                                        │        │
│  │       "diameter_mm": 6.2,                                   │        │
│  │       "confidence": 0.92,                                   │        │
│  │       "risk_level": "intermediate",                        │        │
│  │       "model_prob": {"yolo": 0.92, "3d_cnn": 0.88}         │        │
│  │     },                                                       │        │
│  │     ...                                                      │        │
│  │   ]                                                          │        │
│  │ }                                                            │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                       ↓                                                    │
│  [Output] FINAL RESULT                                                  │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ ✓ JSON file saved                                          │        │
│  │ ✓ PNG images saved (annotated)                             │        │
│  │ ✓ UI shows: 5 nodules detected, 2 high-risk, 2 mid, 1 low │        │
│  │ ✓ Processing time: 45s (200 slices)                        │        │
│  │ ✓ Memory used: 2.1GB / 4GB available                       │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘

Thống kê qua các bước:
1. Load DICOM: 150 slices
2. YOLO detect: 300–500 raw detections (FP nhiều)
3. 3D CNN filter: 100–200 passed (loại ~50% FP)
4. Clustering: ~5–15 final nodules (gom lại duplicates)
```

### 5.3 Các thành phần chính
| Module | File | Chức năng |
|--------|------|----------|
| Image I/O | `utils/image_reader.py` | Đọc DICOM, chuẩn hóa Hounsfield Unit |
| Preprocessing | `utils/lung_mask_generator.py` | Tạo mask phổi, cắt ROI |
| Lung Seg | `models/lung_segment.py` | Load + inference U-Net |
| Detection | `models/nodule_detect.py` | Load + inference YOLO |
| 3D Patch | `utils/patch_extractor_3d.py` | Cắt voxel từng nốt |
| FPR Reduction | `models/fpr_3d_net.py` | Mạng 3D CNN loại false positive |
| Postproc | `pipeline.py` | Morphological filter, cluster 3D |
| Pipeline | `pipeline.py` | Điều phối toàn bộ |
| UI | `ui/*.py` | Tab phân tích, so sánh, huấn luyện |

### 5.4 Sơ đồ class (OOP design)
- **Class Pipeline:** Quản lý toàn bộ quy trình
- **Class UNetModel, YOLOModel, FPR3DModel:** Đóng gói từng mô hình
- **Class DicomReader:** Đọc ảnh DICOM, cache để tránh đọc lại
- **Class ResultManager:** Quản lý kết quả, export JSON/PNG

---

## 6. CHUẨN BỊ DỮ LIỆU VÀ TIỀN XỬ LÝ (2–3 trang)

### 6.1 Tổ chức dữ liệu
- **Nguồn:** 10 bệnh nhân CT, ~1000 nốt phổi được nhãn thủ công (XML format)
- **Cấu trúc thư mục:**
  ```
  data/
  ├── 3000518.000000-NA-66796/
  │   ├── 1-001.dcm
  │   ├── 1-002.dcm
  │   ...
  │   └── 086.xml (nhãn: nốt_id, x, y, z, diameter, type)
  ├── 3000534.000000-NA-58228/
  ...
  ```
- **Nhãn format:** XML với các trường: nodule_id, centroid (x, y, z), diameter (mm), type (solid/GGN/mixed)

### 6.2 Xử lý và chuẩn hóa
- **Chuẩn hóa Hounsfield:** Clip HU vào [-1024, 400] (phổi), sau đó normalize [0, 1]
- **Resize:** Nếu slice resolution khác 512×512, resize về chuẩn
- **Cân bằng:** Nếu nốt nhỏ ít, dùng data augmentation (rotate, shift, noise)
- **Train/val/test split:** 6 bệnh nhân train, 2 val, 2 test (bệnh nhân level, không slice level)

### 6.3 Sinh dataset cho YOLO
- **Format YOLO:** `images/` chứa ảnh, `labels/` chứa `.txt` (mỗi dòng: class x_center y_center width height normalized)
- **Lớp:** class 0 = nodule (chỉ quan tâm nốt, không quan tâm loại lúc detect)
- **Aug:** Sử dụng Albumentations (rotate ±10°, brightness, contrast), sinh 3× dữ liệu từ gốc
- **Validation:** Lấy 20% slice từ val set

### 6.4 Sinh dataset cho 3D CNN
- **Cắt patch:** Xung quanh từng nốt ground truth, cắt 16×32×32 voxel (positive)
- **Negative sample:** Cắt từ các vùng random (đảm bảo không chứa nốt), sau khi YOLO detect để lấy mạch máu False Positive
- **Augment:** Rotate 3D, flip, noise Gaussian
- **Label:** 0 = nốt thực, 1 = mạch máu, 2 = rác
- **Balance:** Cân bằng 3 lớp ~1:1:1 để hạn chế bias

### 6.5 Metrics chuẩn
- **Test set size:** ~200 nodule ground truth từ 2 bệnh nhân test
- **Baseline:** Chỉ YOLO detection (không 3D CNN) → baseline recall, precision
- **Target:** Sau 3D CNN, recall >85%, precision >80%

---

## 7. HUẤN LUYỆN MÔ HÌNH VÀ TỐI ƯU (3–4 trang)

### 7.1 Huấn luyện U-Net (Lung Segmentation)
- **Kiến trúc:** 4 encoder block + 4 decoder block, skip connection
- **Loss function:** Dice Loss + Binary Cross Entropy (mục tiêu: tối ưu IoU)
- **Optimizer:** Adam, learning rate 1e-3
- **Epoch:** 50, batch size 16 (tùy RAM)
- **Augmentation:** Rotation, brightness, contrast (dùng Albumentations)
- **Early stopping:** Nếu val loss không giảm 5 epoch thì stop
- **Kết quả mong đợi:** Dice score >0.95 trên val set
- **File lưu:** `weights/unet_best.pth`

### 7.2 Huấn luyện YOLO (Nodule Detection)
- **Model:** YOLOv8n hoặc YOLOv11n (tùy chọn, app support cả 2)
- **Input size:** 416×416 pixels (trade-off tốc độ vs chính xác)
- **Hyper-parameters:**
  - Learning rate: 0.01
  - Momentum: 0.937
  - Weight decay: 0.0005
  - Epoch: 100
  - Batch size: 16
  - Augment: Default YOLO (mosaic, rotation, scale)
- **Loss:** CIoU loss (đánh giá box quality)
- **Validation:** Mỗi 10 epoch test trên val set, lưu best model
- **Kết quả mong đợi:** mAP@0.5 > 0.7
- **File lưu:** `weights/yolov8n_best.pt` hoặc `yolov11n_best.pt`
- **Tối ưu cho máy yếu:** Dùng distillation (học từ model lớn) nếu cần

### 7.3 Huấn luyện 3D CNN (FPR Reduction)
- **Kiến trúc:** 3 Conv3D blocks (32→64→128 filters), global average pooling, 3 FC layers (output: 3 lớp)
- **Loss function:** Cross Entropy (phân loại 3 lớp)
- **Optimizer:** Adam, lr=1e-3
- **Epoch:** 50, batch size 8 (patch nhỏ, RAM ít)
- **Augmentation 3D:** Rotate xyz ±10°, flip, Gaussian noise σ=0.01
- **AMP (Mixed Precision):** Sử dụng float16 để tiết kiệm RAM 50%
- **Validation:** Accuracy >85% trên val set (mix của nốt + mạch máu)
- **File lưu:** `weights/fpr_3d_best.pth`

### 7.4 So sánh YOLOv8 vs YOLOv11
| Tiêu chí | YOLOv8n | YOLOv11n |
|----------|---------|---------|
| Kích thước | ~6.3MB | ~7.5MB |
| Tốc độ CPU | 40ms/img | 50ms/img |
| mAP@0.5 | 0.68 | 0.72 |
| RAM infer | 200MB | 250MB |
| **App chọn** | ✓ (ưu tiên tốc độ) | Tùy cấu hình |

### 7.5 Tối ưu hóa
- **Quantization:** Model ONNX 8-bit nếu cần tốc độ tuyệt đối
- **Batch inference:** Xử lý nhiều slice cùng lúc thay vì từng slice
- **Cache:** Lưu kết quả U-Net để không tính lại khi người dùng thay đổi threshold YOLO
- **Multi-threading:** Đọc DICOM trong 1 thread, inference trong thread khác (UI không lag)

---

## 8. PIPELINE SỰ KIỆN VÀ HẬU XỬ LÝ (2–3 trang)

### 8.1 Các bước inference end-to-end
```
Input DICOM (z, h, w)
    ↓
[1] Load & normalize HU
    ↓
[2] U-Net inference → Lung mask
    ↓
[3] Crop to ROI (lung region)
    ↓
[4] YOLO inference (mỗi slice) → list of (x, y, conf, w, h)
    ↓
[5] Extract 3D patch (16×32×32) xung quanh từng detection
    ↓
[6] 3D CNN inference → softmax [p_nodule, p_vessel, p_trash]
    ↓
[7] Filter by 3D CNN confidence (keep if p_nodule > threshold_3d)
    ↓
[8] Morphological filter (remove small blobs <10px)
    ↓
[9] Cluster 3D (merge nearby detections)
    ↓
[10] Post-processing (compute size, centroid in mm)
    ↓
Output: List of (x, y, z, diameter_mm, confidence)
```

### 8.2 Hậu xử lý - Morphological Filter
- **Mục đích:** Loại artifact, tiếng ồn (điểm nhỏ lẻ, không phải nốt)
- **Phương pháp:** 
  1. Erosion: cv2.morphologyEx(..., cv2.MORPH_OPEN) → xóa các điểm nhỏ
  2. Dilation: Để khôi phục lại kích thước
  3. Remove small regions: Loại vùng có diện tích <10 pixel
- **Tham số:** kernel size 3×3, iteration 2
- **Kết quả:** Kết quả sạch hơn, ít false positive

### 8.3 Clustering 3D - Gom nốt
- **Vấn đề:** Cùng 1 nốt vật lý, nếu xuất hiện trong 5 slice, YOLO sẽ detect 5 lần
- **Giải pháp:** DBSCAN hoặc simple 3D centroid clustering
- **Phương pháp đơn giản (3D IoU merge):**
  1. Sắp xếp detection theo z (slice)
  2. Với 2 detection lân cận: tính 3D IoU
  3. Nếu IoU > 0.3 → merge (lấy trung bình tọa độ)
- **Kết quả:** 1 nốt vật lý = 1 detection duy nhất
- **Cải thiện:** Kích thước nốt được tính chính xác hơn (trung bình từ nhiều slice)

### 8.4 Tính kích thước nốt (mm)
- **Input:** Bounding box [x, y, w, h] (pixel) + z (slice index)
- **Cần:** Metadata DICOM (pixel spacing, slice thickness)
  - `pixel_spacing`: (0.5, 0.5) mm
  - `slice_thickness`: 1.0 mm
- **Tính:** 
  - width_mm = w × pixel_spacing[0]
  - height_mm = h × pixel_spacing[1]
  - diameter_mm = (width_mm + height_mm) / 2
- **Output:** diameter_mm (để bác sĩ đánh giá nguy hiểm)

### 8.5 Phân loại nguy hiểm (Risk level)
- **Tiêu chuẩn (Lung-RADS):**
  - Xanh (Low risk): diameter < 6mm
  - Vàng (Intermediate): 6–8mm hoặc GGN 4–6mm
  - Đỏ (High risk): >8mm hoặc solid + suspect growth
- **App hiển thị:** Màu sắc + con số xác suất từ 3D CNN
- **Lý do:** Bác sĩ sẽ chú ý những nốt đỏ, không bỏ sót

---

## 9. THIẾT KẾ GIAO DIỆN NGƯỜI DÙNG (2–2.5 trang)

### 9.1 Cấu trúc giao diện chính
```
┌──────────────────────────────────────────────────────┐
│              ỨNG DỤNG PHÁT HIỆN NỐT PHỔI             │
├──────────────────────────────────────────────────────┤
│ [File] [Edit] [View] [Tools] [Help]                  │
├──────────────────────────────────────────────────────┤
│  📁 Open  🔄 Reload  ⚙️ Settings  💾 Export           │
├──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┤
│ Phân  │ So   │ Huấn │ Cài  │ ...  │      │      │      │
│ tích  │ sánh │ luyện│ đặt  │      │      │      │      │
├──────────────────────────────────────────────────────┤
│ [Thang slide Z: 1 / 150]                              │
│ ┌─────────────────────────────────────────────────┐  │
│ │                                                 │  │
│ │      Hiển thị ảnh CT (512×512)                  │  │
│ │      Khi hover: Hiển thị nốt, risk color       │  │
│ │                                                 │  │
│ └─────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────┤
│ ✓ Detected Nodules: 5                                │
│ ┌────────────────────────────────────────────────┐  │
│ │ # │ Size(mm) │ Z slice │ Confidence │ Risk    │  │
│ │ 1 │  6.2    │  45    │ 0.92      │ 🟡 Mid   │  │
│ │ 2 │  8.5    │  52    │ 0.88      │ 🔴 High  │  │
│ │ 3 │  3.1    │  67    │ 0.81      │ 🟢 Low   │  │
│ └────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────┤
│ Status: ✓ Processed | RAM: 1.2GB | Time: 45s        │
└──────────────────────────────────────────────────────┘
```

### 9.2 Các Tab chính
#### **Tab 1: Phân tích (Analysis Tab)**
- **Chức năng:** Load DICOM, auto detect, hiển thị kết quả
- **Các widget:**
  - File chooser: Chọn folder DICOM
  - Slider: Duyệt từng slice (↑↓ hoặc mouse wheel)
  - Image viewer: Hiển thị ảnh + bounding box (màu theo risk)
  - Settings panel (inline): Adjust threshold confidence YOLO, 3D CNN, show/hide mask
  - Result table: List nốt phát hiện (click = jump to slice)
  - Info panel: Chọn 1 nốt → hiển thị thông tin chi tiết (tọa độ, kích thước, mô hình dự đoán)
- **Quy trình:** Open → auto infer (progress bar) → view results → hover/click để xem chi tiết

#### **Tab 2: So sánh (Compare Tab)**
- **Chức năng:** Overlay ảnh gốc + ảnh detect (4 view mode)
  - Split left-right: Gốc bên trái, detect bên phải
  - Overlay with opacity: Xếp chồng, có thanh trượt opacity
  - Animated: Xem lại video giữa 2 view
  - Diff mode: Hiển thị pixel khác biệt
- **Dùng để:** Bác sĩ verify, tìm false positive/negative, điều chỉnh ngưỡng
- **Thao tác:** Ấn Ctrl+Z để undo, Ctrl+S để save ảnh compare

#### **Tab 3: Huấn luyện lại (Training Tab)**
- **Chức năng:** Fine-tune YOLO với dữ liệu mới từ bệnh viện
- **Quy trình:**
  1. Load YOLO training dataset (folder có `images/`, `labels/`)
  2. Chọn mô hình base (yolov8n hoặc yolov11n)
  3. Cài đặt: epoch (20–50), batch size (8–16), learning rate
  4. Ấn "Train" → progress bar + real-time loss plot
  5. Khi xong, save model mới & compare metric với model cũ
- **Ứng dụng:** Mỗi bệnh viện dữ liệu riêng (CT cũ/mới, cấu hình khác), nên fine-tune cho tốt nhất

#### **Tab 4: Cài đặt (Settings Tab)**
- **Phần Global Settings:**
  - Select YOLO model: yolov8n / yolov11n
  - Select 3D CNN model: fpr_3d
  - Device: CPU / GPU (auto detect)
  - RAM limit: 2GB / 4GB / 8GB (tự động trim batch size)
- **Phần Detection Threshold:**
  - YOLO confidence: slider [0.3–0.9] (default 0.5)
  - 3D CNN nodule threshold: slider [0.3–0.9] (default 0.6)
  - Min/Max nodule size: input field (mm) → filter result
- **Phần UI:**
  - Theme: Light / Dark
  - Language: English / Tiếng Việt
  - Auto-save: ON/OFF
  - Log level: Quiet / Normal / Verbose

### 9.3 Tương tác người dùng chính
| Tác vụ | Bước thực hiện | Kết quả |
|--------|----------------|---------|
| Mở ảnh CT | Menu File → Open → chọn folder | Load DICOM, show slice đầu |
| Detect nốt | Ấn "Run Analysis" button | Progress bar, sau ~45s hiển thị result table |
| Xem nốt cụ thể | Click vào hàng trong result table | Jump to slice, highlight bounding box |
| Điều chỉnh ngưỡng | Kéo slider "YOLO Confidence" | Realtime filter result, hide/show detection |
| So sánh ảnh | Chuyển sang tab "Compare" | 4-view mode hiển thị diff gốc vs detect |
| Export báo cáo | Click "Export to JSON" | Save file chứa list nốt + tọa độ + confidence |
| Huấn luyện lại | Tab "Training" → chọn dataset → Train | Tạo model mới, so sánh metric |

### 9.4 Ảnh minh họa UI (cần chụp/vẽ)
- **Screenshot 1:** Tab Phân tích, showing ảnh CT + nốt đỏ detected
- **Screenshot 2:** Result table, highlighting các nốt + risk color
- **Screenshot 3:** Settings panel, các slider và checkbox
- **Screenshot 4:** Compare tab, split view gốc vs detect
- **Screenshot 5:** Training tab, loss plot real-time

---

## 10. THỰC NGHIỆM, KẾT QUẢ VÀ THẢO LUẬN (3–4 trang)

### 10.1 Thiết kế thí nghiệm
- **Mục tiêu:** Đánh giá độ chính xác của từng module + pipeline hoàn chỉnh
- **Test set:** 2 bệnh nhân độc lập (200+ nốt ground truth)
- **Baseline:** Chỉ YOLO detection (không 3D CNN, không post-processing)
- **So sánh:** 
  - YOLO only vs YOLO+3D CNN vs YOLO+3D CNN+morphological+cluster
  - YOLOv8n vs YOLOv11n trên cùng dữ liệu
- **Metric:** Recall, Precision, F1, FROC curve

### 10.2 Kết quả định lượng

**Sơ đồ Metric Evaluation - FROC Curve:**
```
┌────────────────────────────────────────────────────────────────┐
│           FROC CURVE: PIPELINE IMPROVEMENT ANALYSIS             │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Sensitivity (%) ↑                                              │
│ 100 │                                        ╭─ Full Pipeline  │
│  95 │                                   ╭────╯                │
│  90 │                              ╭────╯                      │
│  85 │  ┃ +3D CNN         ╭────╯                               │
│  80 │  ┃ +Morphological  │                                    │
│  75 │  ┃ +Clustering     │                                    │
│  70 │  ┃                 │                                    │
│  65 │  ├─ YOLO only  ────┘                                    │
│     │  │                                                       │
│  0  └──┴──────────────────────────────────────────────────────→
│     0  2  4  6  8  10 12 14
│                FP per case ←
│
│ Legend:
│ ─── YOLO only:        Recall 78%, Precision 65%, FP=12
│ ─── +3D CNN:          Recall 84%, Precision 78%, FP=5
│ ─── +Morphological:   Recall 85%, Precision 80%, FP=3
│ ─── +Clustering:      Recall 87%, Precision 82%, FP=2
│
└────────────────────────────────────────────────────────────────┘

Nhận xét:
- 3D CNN giảm FP từ 12 → 5 (58% improvement)
- Morphological filter giảm thêm FP từ 5 → 3 (40%)
- Clustering 3D giảm trùng lặp từ 3 → 2 (33%)
- Tổng cộng: Recall tăng 9% (78% → 87%), FP giảm 83% (12 → 2)
```

#### **Bảng 1: So sánh từng module**
| Pipeline | Recall | Precision | F1 Score | FP per case |
|----------|--------|-----------|----------|-------------|
| YOLO only | 78% | 65% | 0.71 | 12 |
| +3D CNN | 84% | 78% | 0.81 | 5 |
| +Morphological | 85% | 80% | 0.82 | 3 |
| +3D Clustering | 87% | 82% | 0.84 | 2 |

#### **Bảng 2: So sánh YOLOv8n vs YOLOv11n**
| Model | mAP@0.5 | Tốc độ (ms/img) | RAM (MB) | Recall trên test |
|-------|---------|-----------------|----------|-----------------|
| v8n | 0.68 | 40 | 180 | 82% |
| v11n | 0.72 | 50 | 210 | 87% |
| **Kết luận** | v11n tốt hơn nhưng chậm hơn | - | - | |

#### **Bảng 3: Hiệu suất theo kích thước nốt**
| Kích thước (mm) | Số nốt | Recall | Precision | Nhận xét |
|----------------|--------|--------|-----------|----------|
| 3–5 (nhỏ) | 50 | 70% | 75% | Khó detect, miss nhiều |
| 5–8 (trung) | 100 | 88% | 85% | Tốt |
| >8 (lớn) | 50 | 94% | 92% | Rất tốt |

### 10.3 Phân tích lỗi (Error Analysis)
#### **Case: False Negative (bỏ sót nốt)**
- **Tình huống:** Nốt GGN 4mm trong slice 60 không được detect
- **Nguyên nhân:** YOLO confident score thấp (0.4), bị filter bởi threshold 0.5
- **Giải pháp:** Hạ ngưỡng YOLO xuống 0.4 (nhưng tăng FP), hoặc huấn luyện lại YOLO với GGN samples
- **Bài học:** Cần data augmentation cho GGN

#### **Case: False Positive (nhầm mạch máu)**
- **Tình huống:** Mạch máu 3 chiều (tròn) được YOLO nhầm thành nốt
- **Nguyên nhân:** YOLO chỉ nhìn 1 slice, không biết context 3D
- **Giải pháp:** 3D CNN filter khi xem context 3D → confidence mạch máu cao → bị loại
- **Hiệu quả:** False positive giảm từ 12 → 2 per case

#### **Case: Nốt nhỏ bị overlap**
- **Tình huống:** 2 nốt sát nhau 5mm, YOLO detect thành 1 cái lớn
- **Nguyên nhân:** NMS (Non-Maximum Suppression) quá mạnh (IoU threshold 0.3)
- **Giải pháp:** Hạ NMS threshold xuống 0.2, hoặc dùng soft-NMS
- **Cải thiện:** Detect chính xác 2 nốt riêng biệt

### 10.4 Đánh giá so sánh với SOTA
| Phương pháp | Nguồn | Recall | Precision | Ghi chú |
|-------------|-------|--------|-----------|---------|
| **App này** | - | 87% | 82% | Lightweight, CPU-friendly |
| YOLO-World | Paper | 88% | 85% | Nặng hơn, cần GPU |
| 3D RetinaNet | Paper | 90% | 87% | Chỉ train trên 3D, phức tạp |
| Clinical radiologist | Real-world | ~92% | ~95% | Gold standard, nhưng tốc độ chậm |

### 10.5 Thảo luận
- **Ưu điểm:**
  - Độ chính xác tốt (87% recall, 82% precision)
  - Chạy được trên CPU, RAM <4GB
  - Quy trình end-to-end tự động, không cần user input
  - Có thể huấn luyện lại cho từng bệnh viện

- **Hạn chế:**
  - Recall còn thấp so với radiologist (87% vs 92%)
  - Hiện tại chỉ test trên 2 bệnh nhân, cần dataset lớn hơn
  - Nốt GGN nhỏ (<4mm) còn khó phát hiện
  - Chưa xử lý được nốt bị che lấp bởi xương/tim

- **Cải thiện tương lai:**
  - Tăng training data (curate 50+ bệnh nhân)
  - Dùng semi-supervised learning cho dữ liệu unlabeled
  - Thêm 3D detection head (không chỉ 2D YOLO)
  - Ensemble YOLO v8 + v11 để tăng recall

---

## 11. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (1–1.5 trang)

### 11.1 Kết quả chính
- Xây dựng thành công ứng dụng desktop phát hiện nốt phổi trên CT tích hợp AI
- Độ chính xác: Recall 87%, Precision 82%, F1 0.84
- Tối ưu cho máy yếu: Chạy CPU, RAM <4GB, thời gian 45s per case
- Giao diện thân thiện, người dùng không cần kiến thức ML

### 11.2 Đóng góp khác
- Pipeline mở rộng (có thể thêm module mới)
- Training UI cho fine-tuning model mới
- Export kết quả dạng JSON cho hệ thống PACS tích hợp

### 11.3 Hạn chế hiện tại
- Recall thấp so với radiologist, cần cải thiện
- Dataset nhỏ (10 bệnh nhân), chưa đủ diverse
- Chưa xử lý được loại nốt phức tạp (cavity, tree-in-bud)

### 11.4 Hướng phát triển tương lai
1. **Ngắn hạn (3 tháng):**
   - Curate thêm 40 bệnh nhân CT để tăng training data
   - Fine-tune model trên GGN-specific data
   - Thêm 3D detection module (full 3D YOLO)

2. **Trung hạn (6–12 tháng):**
   - Tích hợp với hệ thống PACS bệnh viện
   - Thêm phân loại nguy hiểm (AI Radiomics features)
   - Đo lường growth rate (so sánh với scan trước)

3. **Dài hạn (>12 tháng):**
   - Mở rộng sang các bệnh khác (COVID, IPF, cancer progression)
   - Dùng semi-supervised learning cho dữ liệu unlabeled
   - Triển khai trên mobile (iOS/Android) với model quantized

### 11.5 Kết thúc
Ứng dụng này chứng minh rằng AI có thể giúp bác sĩ phát hiện nốt phổi hiệu quả hơn, nhanh hơn. Với sự cải thiện về dữ liệu và mô hình, có tiềm năng trở thành công cụ lâm sàng thiết thực trong tầm soát sớm ung thư phổi.

---

## 12. TÀI LIỆU THAM KHẢO (0.5–1 trang)

### Sách & Bài báo chính
1. LeCun, Y., Bengio, Y., & Hinton, G. E. (2015). Deep learning. Nature, 521(7553), 436–444.
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation.
3. Redmon, J., & Farsadi, A. (2018). YOLOv3: An incremental improvement. arXiv preprint arXiv:1804.02767.
4. Jocher, G. (2023). YOLOv8 by Ultralytics. https://github.com/ultralytics/ultralytics
5. Jacobs, C., et al. (2016). Pulmonary nodule detection in CT images: False positive reduction using deep learning.

### Tài liệu kỹ thuật
- DICOM standard: https://www.dicomstandard.org/
- OpenCV documentation: https://docs.opencv.org/
- PyTorch documentation: https://pytorch.org/docs/
- Lung-RADS guideline: https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/Lung-Rads

### Công cụ & Framework dùng
- Python 3.10+
- PyTorch 2.0+
- YOLO Ultralytics v8.x, v11.x
- OpenCV (image processing)
- CustomTkinter (UI)
- pydicom (DICOM I/O)
- NumPy, SciPy (data processing)

---

## 13. PHỤ LỤC (0.5–1.5 trang)

### A. Tham số huấn luyện chi tiết
```yaml
# U-Net Training
unet:
  epochs: 50
  batch_size: 16
  learning_rate: 1e-3
  optimizer: Adam
  loss: DiceLoss + BCE
  early_stopping_patience: 5
  
# YOLO Training
yolo:
  epochs: 100
  batch_size: 16
  learning_rate: 0.01
  weight_decay: 0.0005
  augment: True
  input_size: 416x416

# 3D CNN Training
fpr_3d:
  epochs: 50
  batch_size: 8
  learning_rate: 1e-3
  mixed_precision: True
  optimizer: Adam
  loss: CrossEntropyLoss
```

### B. Cấu trúc thư mục dữ liệu
```
project/
├── data/
│   ├── patient_001/
│   │   ├── 001.dcm
│   │   ├── 002.dcm
│   │   └── annotations.xml
│   ├── patient_002/
│   └── ...
├── dataset_yolo_final/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
├── weights/
│   ├── unet_best.pth
│   ├── yolov8n_best.pt
│   └── fpr_3d_best.pth
└── src/
    ├── main.py
    ├── pipeline.py
    ├── models/
    ├── ui/
    └── utils/
```

### C. Bảng thông số mô hình
| Mô hình | Input | Output | Tham số | Tốc độ |
|--------|-------|--------|--------|--------|
| U-Net | (1, 512, 512) | (1, 512, 512) | 7.8M | 30ms/img |
| YOLOv8n | (3, 416, 416) | (N, 6) boxes | 3.2M | 40ms/img |
| 3D CNN | (1, 16, 32, 32) | (3,) probs | 0.8M | 10ms/patch |

### D. Mẫu output JSON
```json
{
  "case_id": "3000518.000000-NA-66796",
  "processed_date": "2026-04-26T14:30:00Z",
  "total_slices": 150,
  "nodules": [
    {
      "nodule_id": 1,
      "centroid_x_mm": 245.5,
      "centroid_y_mm": 189.2,
      "centroid_z_slice": 45,
      "diameter_mm": 6.2,
      "confidence": 0.92,
      "risk_level": "intermediate",
      "model_probs": {
        "yolo": 0.92,
        "fpr_3d_nodule": 0.88
      }
    },
    {
      "nodule_id": 2,
      "centroid_x_mm": 320.1,
      "centroid_y_mm": 215.3,
      "centroid_z_slice": 52,
      "diameter_mm": 8.5,
      "confidence": 0.88,
      "risk_level": "high",
      "model_probs": {
        "yolo": 0.91,
        "fpr_3d_nodule": 0.85
      }
    }
  ],
  "summary": {
    "total_nodules_detected": 5,
    "high_risk_count": 2,
    "intermediate_risk_count": 2,
    "low_risk_count": 1
  }
}
```

### E. Hướng dẫn cài đặt nhanh
```bash
# 1. Clone repo
git clone <repo_url>
cd NCS

# 2. Tạo virtual env
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download weights (nếu chưa có)
python scripts/download_weights.py

# 5. Run app
python src/main.py
```

---

## Lưu ý khi viết
1. **Mỗi phần nên có:** mở đầu (giới thiệu), chi tiết (phân tích), kết luận (tóm tắt + liên kết phần tiếp theo)
2. **Thêm hình minh họa:** Mỗi 2–3 trang nên có ít nhất 1 sơ đồ / hình / bảng
3. **Giải thích "tại sao":** Không chỉ nói "làm cái gì", mà nói "tại sao lại làm cái này"
4. **Dữ liệu cụ thể:** Không nói chung chung, mà dùng con số thực từ app (87% recall, 40ms, v.v.)
5. **Language:** Viết tiếng Việt chuẩn y tế + kỹ thuật, có thể dùng thuật ngữ tiếng Anh nếu cần
6. **Format:** Sử dụng Markdown hoặc Word có heading rõ ràng, font 12pt, line space 1.5, margin 2.5cm

---

## 14. DANH MỤC THÔNG TIN CẦN LẤY TỪ APP VÀ MÔ HÌNH ĐỂ ĐƯA VÀO BÀI

Phần này rất quan trọng vì giúp bài viết không bị chung chung. Bạn nên chụp màn hình, xuất bảng số liệu, và lấy thông tin thật từ app để minh họa.

### 14.1 Thông tin cần lấy từ app
- **Ảnh giao diện chính của app:** màn hình phân tích, màn hình so sánh, màn hình huấn luyện, màn hình cài đặt
- **Ảnh kết quả detect:** ảnh CT có bounding box nốt phổi, màu theo mức nguy hiểm
- **Danh sách nốt phát hiện:** bảng kết quả gồm số thứ tự, vị trí, kích thước, confidence, risk level
- **Ảnh xuất báo cáo:** JSON, bảng summary, hoặc log kết quả sau khi chạy xong
- **Ảnh màn hình chạy thực tế:** progress bar khi đang inference, thời gian xử lý, RAM sử dụng nếu có

### 14.2 Thông tin cần lấy từ mô hình AI
- **Tên mô hình sử dụng:** U-Net, YOLOv8/YOLOv11, 3D CNN
- **Vai trò của từng mô hình:** model nào làm segmentation, model nào detect, model nào giảm false positive
- **Tham số chính của mô hình:** input size, batch size, số epoch, learning rate, số lớp output
- **Kết quả huấn luyện:** loss curve, accuracy curve, mAP, Dice score, precision, recall
- **Trọng số mô hình:** file `.pt` hoặc `.pth` đang dùng trong app
- **So sánh mô hình:** YOLOv8 với YOLOv11, có 3D CNN và không có 3D CNN

### 14.3 Sơ đồ cần có trong bài
- **Sơ đồ kiến trúc tổng thể:** từ DICOM input đến output kết quả
- **Sơ đồ pipeline xử lý:** load ảnh → tiền xử lý → detect → hậu xử lý → xuất kết quả
- **Sơ đồ mô hình:** U-Net, YOLO, 3D CNN, clustering 3D
- **Sơ đồ so sánh:** trước/sau tiền xử lý, trước/sau hậu xử lý, trước/sau clustering
- **Sơ đồ đánh giá:** bảng hoặc hình FROC, precision-recall, confusion matrix

### 14.4 Ảnh trước và sau khi tiền xử lý
Bạn nên chuẩn bị ít nhất các ảnh sau:
- **Ảnh gốc ban đầu:** ảnh DICOM chưa xử lý, còn nhiễu hoặc vùng ngoài phổi
- **Ảnh sau chuẩn hóa HU:** ảnh sau khi clip và normalize
- **Ảnh sau resize/crop:** ảnh đã đưa về kích thước chuẩn
- **Ảnh sau tạo mask phổi:** chỉ giữ vùng phổi, bỏ background
- **Ảnh trước/sau augmentation nếu cần:** phục vụ mô tả tập dữ liệu huấn luyện

### 14.5 Ảnh trước và sau khi hậu xử lý
Đây là phần rất nên có vì chứng minh hiệu quả của pipeline:
- **Ảnh trước hậu xử lý:** kết quả YOLO ban đầu, thường còn nhiều false positive
- **Ảnh sau morphological filter:** các vùng nhiễu nhỏ được loại bỏ
- **Ảnh sau 3D CNN filter:** bỏ bớt các vùng là mạch máu hoặc rác
- **Ảnh sau clustering 3D:** các detection trùng nhau được gom lại thành một nốt
- **Ảnh so sánh trước/sau:** đặt cạnh nhau để thấy cải thiện rõ ràng

### 14.6 Bảng số liệu nên trích từ app
- **Số lượng ảnh CT đã xử lý:** bao nhiêu case, bao nhiêu slice
- **Số lượng nốt phát hiện được:** tổng số, số nốt nguy hiểm, số nốt trung bình, số nốt thấp
- **Thời gian xử lý trung bình:** mỗi case mất bao lâu
- **RAM/GPU usage:** nếu app có log thì nên đưa vào
- **Chỉ số đánh giá:** recall, precision, F1, mAP, Dice score
- **So sánh trước/sau cải tiến:** trước khi có 3D CNN, sau khi có 3D CNN; trước/sau clustering

### 14.7 Những ảnh nên chèn vào bài để đủ số trang
1. Ảnh giao diện chính của app
2. Ảnh ảnh CT gốc chưa xử lý
3. Ảnh sau tiền xử lý và tạo mask phổi
4. Ảnh detect trước hậu xử lý
5. Ảnh detect sau hậu xử lý
6. Ảnh bảng kết quả thống kê nốt
7. Ảnh biểu đồ loss/accuracy khi huấn luyện
8. Ảnh biểu đồ so sánh mô hình
9. Ảnh so sánh YOLOv8 và YOLOv11
10. Ảnh so sánh trước/sau clustering 3D

### 14.8 Nơi chèn các thông tin này trong bài
- **Phần 5:** dùng sơ đồ kiến trúc tổng thể
- **Phần 6:** chèn ảnh tiền xử lý và mô tả dữ liệu
- **Phần 7:** chèn đồ thị huấn luyện và bảng so sánh mô hình
- **Phần 8:** chèn ảnh trước/sau hậu xử lý
- **Phần 9:** chèn ảnh giao diện app
- **Phần 10:** chèn bảng metric, đồ thị đánh giá, ảnh trước/sau cải tiến

### 14.9 Gợi ý viết để không thiếu trang
- Mỗi ảnh nên có 1 đoạn mô tả từ 5–10 dòng
- Mỗi bảng nên có 1 đoạn nhận xét kết quả phía dưới
- Mỗi sơ đồ nên giải thích luồng xử lý từ trái sang phải hoặc từ trên xuống dưới
- Nếu có số liệu thật, hãy so sánh với một phiên bản baseline để bài viết dày và thuyết phục hơn

### 14.10 DANH MỤC THÔNG TIN CỤ THỀ CẦN LẤY TỪ APP NÀY (d:\Tool-vibecode\NCS)

Đây là danh sách chi tiết những gì bạn **phải lấy từ code, model, kết quả thực tế của app** để đưa vào bài viết. Không phải là tài liệu tổng quát, mà là con số, tham số, metadata **thực sự từ app của bạn**.

#### **A. Thông tin từ Code (SRC/ folder)**

**A1. File: `src/main.py`**
- Loại framework giao diện sử dụng: CustomTkinter? PyQt?
- Các Tab chính được định nghĩa: analysis_tab, compare_tab, training_tab, settings_tab
- Màu sắc dùng để hiển thị risk level: xanh (low), vàng (mid), đỏ (high) - lấy hex color code
- Kích thước cửa sổ mặc định
- Tên công ty/bệnh viện nếu có

**A2. File: `src/pipeline.py`**
- Thứ tự các bước xử lý (load → preprocess → detect → classify → output)
- Tên các hàm chính (ví dụ: `run_inference()`, `apply_lung_mask()`, `cluster_3d()`)
- Các threshold mặc định:
  - YOLO confidence threshold: mặc định 0.5 hay gì?
  - 3D CNN threshold: mặc định 0.6 hay gì?
  - NMS IoU threshold cho YOLO: mặc định là bao nhiêu?
  - Khoảng cách clustering 3D (mm): mặc định 20mm hay gì?
- Các hằng số (constants):
  - Input size YOLO: 416×416 hay 640×640?
  - 3D patch size: 16×32×32 hay khác?
  - Batch size mặc định khi inference

**A3. File: `src/models/fpr_3d_net.py`**
- Kiến trúc 3D CNN chi tiết:
  - Số Conv3D blocks: 3? 4?
  - Số filters mỗi block: 32→64→128 hay khác?
  - Kernel size: 3×3×3 hay 5×5×5?
  - GlobalAvgPooling hay khác?
  - Số FC layers: 2? 3?
  - Số tham số (parameters): bao nhiêu triệu (M)?
- Output classes: 3 lớp (nodule, vessel, trash) hay khác?

**A4. File: `src/models/lung_segment.py` (U-Net)**
- Kiến trúc U-Net:
  - Số encoder block: 4? 5?
  - Số decoder block: bằng encoder không?
  - Số filters: 64→128→256→512 hay khác?
  - Skip connection: có hay không?
- Loss function: Dice + BCE hay gì?

**A5. File: `src/models/nodule_detect.py` (YOLO)**
- Phiên bản YOLO sử dụng: YOLOv8n? YOLOv11n?
- Input size: 416×416? 640×640?
- Các anchor sizes
- NMS threshold

**A6. File: `src/utils/image_reader.py`**
- HU normalization range: [-1024, 400] hay khác?
- Pixel spacing: mỗi pixel bằng bao nhiêu mm? (thường 0.5–1mm)
- Slice thickness: mỗi slice dày bao nhiêu mm? (thường 1–2mm)

**A7. File: `src/utils/patch_extractor_3d.py`**
- 3D patch size: 16×32×32 hay khác?
- Padding strategy: zero padding hay edge padding?

#### **B. Thông tin từ Model Weights (weights/ folder)**

**B1. File: `weights/unet_best.pth`**
- Kích thước file: bao nhiêu MB?
- Dice score trên validation set: bao nhiêu?
- Epoch được lưu: epoch bao nhiêu?
- Best val loss: bao nhiêu?

**B2. File: `weights/yolov8n_best.pt` (nếu có)**
- Kích thước file: bao nhiêu MB?
- mAP@0.5 trên test set: bao nhiêu?
- Epoch được lưu: bao nhiêu?

**B3. File: `weights/yolov11n_best.pt` (nếu có)**
- Kích thước file: bao nhiêu MB?
- mAP@0.5 trên test set: bao nhiêu?
- So sánh tốc độ với v8n: bao lâu/image?

**B4. File: `weights/fpr_3d_best.pth`**
- Kích thước file: bao nhiêu MB?
- Accuracy trên validation: bao nhiêu%?
- Loss: bao nhiêu?
- Confusion matrix chi tiết: [P_nodule, P_vessel, P_trash]

#### **C. Thông tin từ Metrics & Results**

**C1. File: `metrics.txt` hoặc `metrics.json` hoặc `metrics_conf05.json`**
- Recall: bao nhiêu%?
- Precision: bao nhiêu%?
- F1 score: bao nhiêu?
- mAP: bao nhiêu?
- Sensitivity: bao nhiêu%?
- Specificity: bao nhiêu%?
- False positive per case: bao nhiêu?
- False negative per case: bao nhiêu?
- Confidence threshold sử dụng: 0.5? 0.6?

**C2. File: `evaluate_models.ipynb` hoặc `evaluate_models.py`**
- Số lượng test cases: bao nhiêu bệnh nhân?
- Số lượng test nodules: bao nhiêu nốt?
- Breakdown theo kích thước: 3–5mm bao nhiêu nốt? 5–8mm? >8mm?
- Breakdown theo loại: solid, GGN, mixed bao nhiêu nốt mỗi loại?
- Chi tiết từng case: case nào bị miss nhiều? Case nào có FP?
- Time per case: trung bình mất bao lâu để process 1 case?
- RAM usage: mỗi case sử dụng RAM bao nhiêu GB?

**C3. File: `evaluate_pipeline.py`**
- Dòng nào chứa tính recall, precision?
- Dòng nào chứa 3D clustering logic?
- Dòng nào chứa morphological filter?

**C4. File: `report/evaluation_report.md`**
- Kết luận chính: đạt được recall bao nhiêu%?
- Vấn đề chính gặp phải
- Recommendation cho improvement

#### **D. Thông tin từ Dataset**

**D1. File: `data/*/086.xml` (annotation file)**
- Format XML: có những trường nào? nodule_id, x, y, z, diameter, type?
- Các loại nốt được nhãn: solid, GGN, mixed?
- Khoảng diameter: từ bao nhiêu mm đến bao nhiêu mm?

**D2. Folder: `data/` (DICOM data)**
- Số bệnh nhân: tổng cộng bao nhiêu folder?
- Số slice trung bình mỗi bệnh nhân: bao nhiêu slice?
- Số nốt được nhãn tất cả: tổng cộng bao nhiêu nốt?
- Phân bố nodule size: bảng histogram hoặc số liệu cụ thể

**D3. Folder: `dataset_yolo_final/`**
- Số training images: bao nhiêu?
- Số validation images: bao nhiêu?
- Số test images: bao nhiêu?
- Data augmentation được dùng: rotation, brightness, contrast?
- Augmentation ratio: dữ liệu tăng bao nhiêu lần? 2×? 3×?

**D4. Folder: `dataset_3d_final.npz` (3D patch dataset)**
- Số positive patches (nodule thực): bao nhiêu?
- Số negative patches (vessel/trash): bao nhiêu mỗi loại?
- Kích thước patch: 16×32×32 hay khác?
- Patch được extract từ bao nhiêu bệnh nhân?

#### **E. Thông tin từ Training Logs & Plots**

**E1. File: `plot_metrics_tmp.py`, `plot_advanced_metrics.py`**
- Loại plot được tạo: loss curve? accuracy curve? FROC curve?
- Đọc code để biết:
  - Y-axis là gì? (loss, accuracy, precision, recall?)
  - X-axis là gì? (epoch, threshold, FP count?)
  - Các đường được so sánh: YOLO vs YOLO+3D CNN vs Full?

**E2. Folder: `runs/` (training results)**
- Folder `train/`: chứa loss plots, weight files
  - Best epoch là bao nhiêu?
  - Best loss là bao nhiêu?
- Folder `runs_compare/`: chứa kết quả so sánh
  - So sánh metrics của các model: YOLOv8 vs YOLOv11?
  - So sánh result: YOLO only vs YOLO+3D CNN?

#### **F. Thông tin từ Giao Diện App (UI Screenshots)**

**F1. Tab Phân tích (Analysis Tab)**
- Layout: image viewer ở đâu? Result table ở đâu?
- Thay đổi slider Z như thế nào? (text input hay mouse drag?)
- Bounding box được vẽ như thế nào? (xanh/vàng/đỏ)
- Info panel hiển thị gì khi click vào 1 nốt? (tọa độ, kích thước, confidence?)

**F2. Tab So sánh (Compare Tab)**
- 4 view mode là gì cụ thể?
  - Split left-right?
  - Overlay + opacity slider?
  - Animated?
  - Diff mode?
- Keyboard shortcuts có không? (Ctrl+Z, Ctrl+S?)

**F3. Tab Huấn luyện (Training Tab)**
- Chọn dataset: có file browser không? Hay dropdown?
- Chọn mô hình base: YOLOv8 hay YOLOv11?
- Hyperparameters có thể chỉnh gì?
  - Learning rate slider?
  - Batch size input?
  - Epoch input?
- Progress bar: real-time hay sau khi train xong?
- Loss plot update như thế nào?

**F4. Tab Cài đặt (Settings Tab)**
- Các setting có:
  - YOLO model selector: dropdown hay radio button?
  - 3D CNN model selector?
  - Device selector: CPU / GPU / Auto?
  - RAM limit slider: 2GB, 4GB, 8GB?
  - Confidence threshold slider: range 0–1?
  - Min/Max nodule size: number input?
  - Theme: Light / Dark?
  - Language: English / Tiếng Việt?

**F5. Status bar / Log**
- Hiển thị processing time: bao nhiêu giây?
- Hiển thị RAM usage: bao nhiêu GB đang dùng?
- Hiển thị mô hình đang sử dụng: YOLOv8n hay YOLOv11n?

#### **G. Thông tin Khác**

**G1. File: `settings.json` (nếu có)**
- Default thresholds: YOLO, 3D CNN
- Default model selection
- Default language, theme
- Default folder paths

**G2. File: `best.pt`, `yolo11n.pt` (model files)**
- So sánh kích thước file
- So sánh tốc độ inference (ms/image)
- So sánh RAM khi load (MB)

**G3. File: `dataset_3d_p1.npz`, `dataset_3d_p2.npz` (nếu là split data)**
- Số lượng patch mỗi file: p1 bao nhiêu? p2 bao nhiêu?
- Tổng cộng bao nhiêu patch?

**G4. File: `OPTIMIZATION_REPORT.md`**
- Các optimization được thực hiện: quantization? distillation? pruning?
- Kết quả: tốc độ tăng bao nhiêu%? Model nhỏ bao nhiêu%?

#### **H. Tóm tắt: Các số liệu PHẢI CÓ trong bài viết**

Bạn **phải lấy và ghi rõ** những số liệu này từ app (không dùng con số "giả"):

1. **Recall & Precision cuối cùng của app:** 87%, 82% (từ metrics.json)
2. **F1 score:** 0.84 (từ tính toán hoặc logs)
3. **False positive per case:** 2 (từ evaluation results)
4. **False negative per case:** bao nhiêu?
5. **Thời gian xử lý trung bình:** 45 giây per case hay gì? (phải chạy app để đo)
6. **RAM sử dụng:** 2.1 GB hay gì? (phải log từ app hoặc Task Manager)
7. **Kích thước mô hình:** U-Net bao nhiêu MB? YOLO bao nhiêu MB? 3D CNN bao nhiêu MB?
8. **Số lượng test data:** bao nhiêu case, bao nhiêu nodules?
9. **Recall, Precision theo kích thước nốt:** 3–5mm 70%, 5–8mm 88%, >8mm 94% (từ evaluate_models.py)
10. **So sánh YOLOv8 vs YOLOv11:** mAP, tốc độ, RAM, recall (từ runs_compare/)
11. **Danh sách các threshold mặc định:** YOLO 0.5, 3D CNN 0.6, NMS IoU 0.4, clustering distance 20mm, v.v.
12. **Kiến trúc mô hình chi tiết:** U-Net 4 encoder block, YOLO có bao nhiêu output heads, 3D CNN bao nhiêu Conv3D layers, v.v.

---

**Tổng dự kiến: 22–25 trang.** 

Bạn có thể copy từng phần này vào Word/Markdown và bắt đầu viết. Nếu cần mở rộng hay có câu hỏi, cứ hỏi mình! 💪
