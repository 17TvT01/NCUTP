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
│   ├── main.py         # Điểm khởi chạy App Desktop (Giao diện CustomTkinter)
│   ├── models/         # Chứa code logic load các model AI (UNet, YOLO, Mini UNet) 
│   ├── utils/          # Chứa các hàm hỗ trợ (image_reader, prepocessing, postprocessing)
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
* **`main.py`**: Chứa giao diện chính của App Desktop sử dụng CustomTkinter. Có chức năng tải ảnh và gọi luồng AI.
* **`models/`**: Thư mục chứa các module tải và chạy model Deep Learning cụ thể.
* **`utils/`**: Thư mục chứa các file code trợ trợ (xử lý hình ảnh, đọc file dicom/nifti).
* **`weights/`**: Thư mục dùng để chứa các file trọng số pre-trained tải từ trên mạng xuống.

---

*(Tài liệu này sẽ được trợ lý AI chủ động cập nhật liên tục mỗi khi có file, thư mục hoặc module mới được tạo/chỉnh sửa trong dự án).*
