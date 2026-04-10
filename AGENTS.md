# Cấu Trúc Dự Án & Chức Năng Các File

Tài liệu này ghi lại toàn bộ cấu trúc thư mục của dự án và tác dụng Cụ thể của từng file để tiện cho việc theo dõi, quản lý và phát triển. 

---

## 🎯 Quy ước phát triển (Coding Conventions)

1. **Phân chia ý tưởng và mã nguồn:**
   * Các file tài liệu, ý tưởng (idea), phân tích sẽ được lưu trữ trong thư mục `idea/`.
   * Toàn bộ mã nguồn (source code) của dự án sẽ được đặt trong thư mục `src/`.
2. **Nguyên tắc thiết kế Code / Component:**
   * **Single Responsibility:** Mỗi file chỉ đảm nhiệm **1 chức năng duy nhất** (1 component / 1 module).
   * **Giới hạn kích thước:** Mỗi file code **không được vượt quá 200 dòng code**. Điều này giúp giữ cho mọi thứ nhỏ gọn, dễ đọc, dễ bảo trì và dễ dàng fix bug sau này. Nếu file có nguy cơ vượt qua giới hạn này, lập tức refactor để tách ra các file nhỏ hơn.

---

## 📂 Cấu trúc thư mục tổng quan

```text
.
├── idea/
│   └── app_ai_plan.md  # Kế hoạch chi tiết phát triển AI khoanh nốt phổi (máy yếu)
├── src/
│   ├── main.py         # Điểm khởi chạy App Desktop, chứa cấu hình gốc và gọi các Tab
│   ├── pipeline.py     # Lõi hệ thống, kết hợp các model (U-Net, YOLO, 3D CNN) và bộ lọc
│   ├── train_fpr_3d.py # Script huấn luyện mạng 3D CNN phân loại nốt phổi và mạch máu
│   ├── data_prep_app.py # UI Độc lập sinh dữ liệu YOLO hàng loạt từ các folder DICOM
│   ├── models/         # Chứa code logic load các model AI (UNet, YOLO, 3D CNN)
│   │   └── fpr_3d_net.py   # Kiến trúc mạng Lightweight 3D CNN chống dương tính giả
│   ├── ui/             # Thư mục chứa giao diện các Tab (chia nhỏ để tuân thủ quy tắc 200 dòng)
│   │   ├── analysis_tab.py # Giao diện tab Phân tích DICOM và Hiển thị kết quả AI
│   │   ├── compare_tab.py # Giao diện tab So sánh ảnh chưa đánh dấu và ảnh đã đánh dấu
│   │   ├── settings_panel.py # Widget khung Thiết lập tham số AI (ngưỡng, voxel, FPR)
│   │   └── training_tab.py # Giao diện tab Huấn luyện lại mô hình YOLO
│   ├── utils/          # Chứa các hàm hỗ trợ (image_reader, prepocessing, postprocessing)
│   │   └── patch_extractor_3d.py # Code trích xuất khối 3D Voxel (16x32x32) từ DICOM
│   └── weights/        # Nơi lưu trữ file trọng số (.pt, .pth)
└── AGENTS.md
```

---

## 📝 Chi tiết chức năng các thành phần

### Thư mục gốc (`/`)

* **`AGENTS.md`**: File hệ thống dùng để theo dõi tổng quan kiến trúc, danh sách các file/thư mục và chức năng tương ứng của từng thành phần trong dự án. (Chính là file này).

### Thư mục `idea/`
* **`app_ai_plan.md`**: Bản kế hoạch, phân tích luồng xử lý và cấu trúc kiến trúc kỹ thuật của dự án AI khoanh nốt phổi, được tối ưu cho máy cấu hình yếu.

### Thư mục `src/`
Thư mục gốc chứa mã nguồn. Các file trong này phải tuân thủ nghiêm ngặt quy tắc mỗi file 1 chức năng, tối đa 200 dòng code.
* **`main.py`**: Chứa giao diện chính (khung Window CustomTkinter) và import các giao diện con (Tab).
* **`pipeline.py`**: Lõi nhận diện chính (Pipeline). Tích hợp U-Net (Lọc phổi), YOLO (Cắt nốt), Morphological Filter (Lọc Hình học) và 3D CNN (Hậu xử lý FPR loại rác).
* **`train_fpr_3d.py`**: Script tích hợp chạy huấn luyện Mạng 3D CNN sử dụng AMP (Mixed Precision) giảm tải RAM.
* **`train_unet.py`**: Script huấn luyện mạng liên quan việc tạo mask (chẳng hạn U-Net).
* **`train_compare_yolo.py`**: Script huấn luyện, so sánh YOLO (ví dụ giữa v8 và v11) và đánh giá chi tiết (Recall, mAP).
* **`data_prep_app.py`**: Giao diện Độc lập (Standalone UI) giúp tự động quét nhiều folder DICOM/XML để sinh và tăng cường dữ liệu huấn luyện hàng loạt.
* **`data_prep_cli.py`**: Công cụ chạy bằng dòng lệnh (CLI) dùng cho việc build và load dữ liệu tự động.
* **`models/`**: Thư mục chứa logic khởi tạo và inference cho các model:
  * `fpr_3d_net.py`: Mạng 3D CNN siêu nhẹ chống False Positives.
  * `lung_segment.py`: Cơ chế load model nhận diện màng phổi.
  * `nodule_detect.py`: Code bọc cho phần load model YOLO nhận diện nốt.
  * `nodule_segment.py`: Module hỗ trợ phân vùng chi tiết nốt phổi.
  * `trainer.py`: Chứa class/trainer để huấn luyện (để tái sử dụng loop train chung).
* **`ui/`**: Thư mục component UI:
  * Gồm nhiều tab chính như `analysis_tab.py`, `compare_tab.py`, `settings_panel.py`, `training_tab.py`.
  * `image_viewer.py`: Thành phần hiển thị hình ảnh Dicom / Render kết quả.
  * `result_tree.py`: Bảng hiển thị danh sách các nốt phát hiện được ở dạng TreeView.
* **`utils/`**: Các hàm tiện ích bổ trợ:
  * `patch_extractor_3d.py`: Cắt các khối 3D Voxel (16x32x32) từ Dicom.
  * `cluster_3d.py`: Gom cụm nốt phổi dựa trên các mặt cắt lân cận sinh ra từ file 2D (nhằm build hệ tọa độ 3D).
  * `image_reader.py`: Logic mở/đọc file DICOM chuẩn y tế và ảnh thông thường.
  * `lung_mask_generator.py`: Tự động nội suy tạo phổi Mask / cắt bỏ background thừa.
  * `data_prep.py`: Các script nhỏ xử lý mảng data tĩnh.
* **`weights/`**: Thư mục dùng để chứa các file trọng số pre-trained tải từ trên mạng xuống.

---

*(Tài liệu này sẽ được trợ lý AI chủ động cập nhật liên tục mỗi khi có file, thư mục hoặc module mới được tạo/chỉnh sửa trong dự án).*
