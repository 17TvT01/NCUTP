# Ý Tưởng Phát Triển Ứng Dụng AI Khoanh Nốt Phổi (Máy Yếu)

Tài liệu này trình bày chi tiết kế hoạch và cấu trúc để phát triển một ứng dụng AI bằng Python có khả năng khoanh vùng nốt phổi từ ảnh CT scan, được tối ưu hóa đặc biệt cho các máy tính có cấu hình yếu (GPU yếu hoặc chỉ có CPU).

## 1. Tổng Quan Pipeline AI Đề Xuất

Để đảm bảo khả năng chạy trên máy yếu mà vẫn giữ được độ chính xác, chúng ta sẽ áp dụng **Pipeline 3 bước tuần tự** thay vì dùng một model khổng lồ xử lý từ A-Z.

**Luồng xử lý (Data Flow):**
1. **Đầu vào:** Ảnh CT Scan định dạng DICOM hoặc NIfTI, hoặc ảnh 2D (PNG/JPG).
2. **Bước 1: Tách Phổi (Lung Segmentation)**
   - **Model:** U-Net 2D (nhẹ, chuyên dụng cho ảnh y tế).
   - **Mục đích:** Loại bỏ các phần không phải phổi (xương, mỡ, nội tạng khác) để giảm nhiễu và thu hẹp không gian tìm kiếm.
3. **Bước 2: Phát Hiện Nốt Phổi (Nodule Detection)**
   - **Model:** YOLOv8n (n là bản nano, cực nhẹ và cực nhanh).
   - **Mục đích:** Tìm các vị trí nghi ngờ có nốt phổi (Bounding box) trong vùng phổi đã được tách ra ở Bước 1.
4. **Bước 3: Khoanh Vùng Nốt (Nodule Segmentation)**
   - **Thao tác:** Từ Bounding box của Bước 2, cắt (crop) ra các vùng nhỏ (patch).
   - **Model:** Mini U-Net 2D chạy trên các vùng nhỏ này.
   - **Mục đích:** Tạo mask chính xác bao quanh nốt phổi.
5. **Đầu ra:** Ảnh kết quả với vùng phổi được xác định và các nốt phổi được highlight (vẽ viền hoặc tô màu mờ).

## 2. Lựa Chọn Công Nghệ (Tech Stack)

*   **Ngôn ngữ:** Python 3.9+
*   **Deep Learning Framework:** `PyTorch` (dễ tùy biến, tài liệu phong phú, hỗ trợ tốt YOLOv8).
*   **Xử lý ảnh Y tế:** `SimpleITK`, `pydicom`, `nibabel` (để đọc file CT).
*   **Xử lý ảnh cơ bản:** `OpenCV`, `Pillow`, `NumPY`.
*   **Model Detection:** Thư viện `ultralytics` (để dùng YOLOv8).
*   **Tạo Giao Diện (GUI):**
    *   *Option 1 (Web nhẹ):* `Streamlit` hoặc `Gradio` (Cực kỳ dễ code, code ngắn, đủ dùng để up ảnh và xem kết quả).
    *   *Option 2 (App Desktop):* `PyQt5` hoặc `CustomTkinter` (Cần tốn công sức code UI hơn nhưng chạy local mượt mà).
    *   *Đề xuất ban đầu:* Dùng `Gradio` hoặc `Streamlit` để prototype nhanh nhất.

## 3. Kiến Trúc Mã Nguồn (Source Code Architecture)

Dự án sẽ tuân thủ quy tắc chức năng đơn lẻ (Single Responsibility) và giới hạn kích thước file, được đặt tại thư mục `src/`.

### Dự kiến cấu trúc thư mục `src/`:

```text
src/
├── main.py                 # (GUI) Điểm bắt đầu của ứng dụng, chứa giao diện chính
├── pipeline.py             # Lắp ráp 3 model lại thành 1 luồng xử lý hoàn chỉnh
├── models/
│   ├── lung_segment.py     # Code load và chạy model U-Net tách phổi
│   ├── nodule_detect.py    # Code load và chạy model YOLOv8n phát hiện nốt
│   └── nodule_segment.py   # Code load và chạy model Mini U-Net khoanh vùng nốt
├── utils/
│   ├── image_reader.py     # Hàm đọc các loại ảnh DICOM, NIfTI, PNG...
│   ├── prepocessing.py     # Hàm chuẩn hóa ảnh (resize, normalize) trước khi đưa vào model
│   └── postprocessing.py   # Hàm vẽ box, vẽ mask lên ảnh gốc để hiển thị
└── weights/                # Nơi chứa các file trọng số (.pt, .pth) của các model
```

## 4. Kế Hoạch Triển Khai (Roadmap)

### Phase 1: Chuẩn bị Inference (Chạy thử model có sẵn)
Thay vì tự train (do máy yếu), chúng ta sẽ tìm kiếm các pre-trained weights (trọng số đã huấn luyện sẵn) trên các tập dữ liệu như LUNA16 hoặc LIDC-IDRI.
1. Viết các hàm cơ bản đọc ảnh DICOM/NIfTI.
2. Tìm và load Pre-trained U-Net cho Lung Segmentation.
3. Tìm và load Pre-trained YOLOv5n/YOLOv8n cho Nodule Detection.
4. Tìm và load Pre-trained cho Nodule Segmentation.
5. Kết nối chúng thành một `pipeline.py` chạy trên terminal trước.

### Phase 2: Refine Code & Xây Dựng GUI
1. Refactor các đoạn code vượt quá quá quy định (200 dòng).
2. Xây dựng giao diện Web App bằng Gradio hoặc Streamlit để người dùng upload ảnh.
3. Tích hợp `pipeline.py` vào GUI.

### Phase 3: Tối Ưu Hóa (Optimization)
1. Thêm tính năng cache model vào RAM để lần dự đoán sau không phải load lại.
2. Ép kiểu dữ liệu (Quantization) về FP16 để model chạy nhanh hơn trên CPU/GPU yếu.

---
*Tài liệu này được định hướng chuyên sâu để xây dựng AI y tế hiệu năng cao.*
