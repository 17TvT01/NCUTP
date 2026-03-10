import customtkinter as ctk
from tkinter import filedialog, ttk
from PIL import Image, ImageDraw
import os
import numpy as np
from utils.image_reader import load_dicom_series
from pipeline import AIPipeline

# Cấu hình giao diện cơ bản của CustomTkinter
ctk.set_appearance_mode("Dark")  # Chế độ tối
ctk.set_default_color_theme("blue")  # Chủ đề màu xanh

class LungNoduleApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Cấu hình cửa sổ chính
        self.title("Lung Nodule Assistant")
        self.geometry("1100x750")
        self.minsize(900, 600)
        
        # Biến lưu trữ dữ liệu
        self.ct_images = []
        self.current_slice_index = 0
        self.analysis_results = {} # Lưu kết quả phân tích theo Slice Index
        
        # Khởi tạo AI Pipeline
        self.ai_pipeline = AIPipeline(device='cpu')
        
        self.setup_ui()

    def setup_ui(self):
        # Tạo TabView chứa 2 chế độ
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=15, pady=0)
        
        self.tabview.add("Phân tích")
        self.tabview.add("Huấn luyện AI")
        
        # --- TAB 1: PHÂN TÍCH ---
        self.main_frame = self.tabview.tab("Phân tích")

        # 1. KHU VỰC: NGUỒN DỮ LIỆU (Có khung)
        # Sử dụng CTkFrame với border_width để tạo hiệu ứng "LabelFrame / GroupBox"
        self.data_group = ctk.CTkFrame(self.main_frame, border_width=1, border_color="#555555", fg_color="#2b2b2b")
        self.data_group.pack(fill="x", pady=(10, 5), ipadx=5, ipady=5)
        
        # Tiêu đề khung mô phỏng GroupBox
        self.lbl_data_title = ctk.CTkLabel(self.data_group, text=" Nguồn dữ liệu & AI ", text_color="#aaaaaa")
        self.lbl_data_title.place(x=10, y=-12) 
        
        # Frame phụ chứa nội dung bên trong khung Nguồn dữ liệu (tránh đè lên title)
        self.data_content = ctk.CTkFrame(self.data_group, fg_color="transparent")
        self.data_content.pack(fill="x", padx=10, pady=(15, 5))

        ctk.CTkLabel(self.data_content, text="Thư mục DICOM / ZIP / MHD").grid(row=0, column=0, padx=(0, 10), pady=5, sticky="w")
        self.entry_dicom_path = ctk.CTkEntry(self.data_content, state="readonly")
        self.entry_dicom_path.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        self.btn_browse = ctk.CTkButton(self.data_content, text="Chọn Thư Mục...", width=100, fg_color="#444444", hover_color="#555555", command=self.browse_dicom_folder)
        self.btn_browse.grid(row=0, column=2, padx=(10, 0), pady=5)
        
        # --- TAB TẢI MODEL YOLO ---
        ctk.CTkLabel(self.data_content, text="Mô hình YOLO (.pt)").grid(row=1, column=0, padx=(0, 10), pady=5, sticky="w")
        self.entry_model_path = ctk.CTkEntry(self.data_content)
        self.entry_model_path.insert(0, "yolov8n.pt")
        self.entry_model_path.configure(state="readonly")
        self.entry_model_path.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        self.btn_browse_model = ctk.CTkButton(self.data_content, text="Đổi Model AI...", width=100, fg_color="#444444", hover_color="#555555", command=self.browse_model_file)
        self.btn_browse_model.grid(row=1, column=2, padx=(10, 0), pady=5)
        
        self.data_content.columnconfigure(1, weight=1)

        # 2. NÚT PHÂN TÍCH CHÍNH GIỮA
        self.btn_analyze = ctk.CTkButton(self.main_frame, text="Phân tích", fg_color="#3b3b3b", hover_color="#4b4b4b", command=self.run_analysis)
        self.btn_analyze.pack(fill="x", pady=10)
        
        self.lbl_status = ctk.CTkLabel(self.main_frame, text="Phát hiện 0 nốt.", anchor="w")
        self.lbl_status.pack(fill="x", pady=(0, 5))

        # 3. KHU VỰC DƯỚI: KẾT QUẢ VÀ HÌNH ẢNH (Chia 2 cột)
        self.bottom_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.bottom_frame.pack(fill="both", expand=True)
        self.bottom_frame.columnconfigure(0, weight=1)
        self.bottom_frame.columnconfigure(1, weight=1)
        self.bottom_frame.rowconfigure(0, weight=1)

        # 3.1 CỘT TRÁI: KHUNG KẾT QUẢ (Có khung viền)
        self.result_group = ctk.CTkFrame(self.bottom_frame, border_width=1, border_color="#555555", fg_color="#2b2b2b")
        self.result_group.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        self.lbl_result_title = ctk.CTkLabel(self.result_group, text=" Kết quả ", text_color="#aaaaaa")
        self.lbl_result_title.place(x=10, y=-12)

        self.result_content = ctk.CTkFrame(self.result_group, fg_color="transparent")
        self.result_content.pack(fill="both", expand=True, padx=10, pady=(15, 10))
        
        # Style cho Treeview (Bảng danh sách các nốt)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#2b2b2b", foreground="#cccccc", fieldbackground="#2b2b2b", borderwidth=0)
        style.configure("Treeview.Heading", background="#333333", foreground="white", borderwidth=0)
        style.map("Treeview", background=[("selected", "#1f538d")])

        self.tree = ttk.Treeview(self.result_content, columns=("ID", "Voxel", "Z", "Y", "X", "Malig", "Color"), show="headings", height=15)
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=50, anchor="center")
        self.tree.pack(fill="both", expand=True)

        # 3.2 CỘT PHẢI: HÌNH ẢNH VÀ SLIDER
        self.image_frame = ctk.CTkFrame(self.bottom_frame, fg_color="#2b2b2b", border_width=1, border_color="#555555")
        self.image_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        self.image_label = ctk.CTkLabel(self.image_frame, text="")
        self.image_label.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.slider_frame = ctk.CTkFrame(self.image_frame, fg_color="transparent")
        self.slider_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.lbl_slice = ctk.CTkLabel(self.slider_frame, text="Lát cắt: 0", width=80)
        self.lbl_slice.pack(side="left", padx=5)
        
        self.slice_slider = ctk.CTkSlider(self.slider_frame, from_=0, to=1, command=self.on_slider_change)
        self.slice_slider.pack(side="left", fill="x", expand=True, padx=5)
        self.slice_slider.set(0)
        self.slice_slider.configure(state="disabled")
        
            # Binds event click vào bảng kết quả
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)
        
        # Gọi hàm tạo UI cho Tab 2
        self._setup_train_tab()

    def _setup_train_tab(self):
        """ Tạo giao diện cho Tab Huấn Luyện AI """
        self.train_frame = self.tabview.tab("Huấn luyện AI")
        
        # 1. KHU VỰC CẤU HÌNH DATASET
        self.train_config_group = ctk.CTkFrame(self.train_frame, border_width=1, border_color="#555555", fg_color="#2b2b2b")
        self.train_config_group.pack(fill="x", pady=10, ipadx=5, ipady=5)
        
        ctk.CTkLabel(self.train_config_group, text=" Cấu hình Huấn luyện ", text_color="#aaaaaa").place(x=10, y=-12)
        
        config_content = ctk.CTkFrame(self.train_config_group, fg_color="transparent")
        config_content.pack(fill="x", padx=10, pady=(15, 5))
        
        # Chọn file data.yaml
        ctk.CTkLabel(config_content, text="Dataset (data.yaml)").grid(row=0, column=0, padx=(0, 10), pady=10, sticky="w")
        self.entry_yaml_path = ctk.CTkEntry(config_content, state="readonly")
        self.entry_yaml_path.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        config_content.columnconfigure(1, weight=1)
        
        self.btn_browse_yaml = ctk.CTkButton(config_content, text="Chọn file...", width=80, fg_color="#444444", hover_color="#555555", command=self.browse_yaml_file)
        self.btn_browse_yaml.grid(row=0, column=2, padx=(10, 0), pady=10)
        
        # Tham số Training (Epochs & Batch Size)
        param_frame = ctk.CTkFrame(config_content, fg_color="transparent")
        param_frame.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 10))
        
        ctk.CTkLabel(param_frame, text="Epochs:").pack(side="left", padx=(0, 5))
        self.entry_epochs = ctk.CTkEntry(param_frame, width=60)
        self.entry_epochs.insert(0, "100")
        self.entry_epochs.pack(side="left", padx=(0, 20))
        
        ctk.CTkLabel(param_frame, text="Batch Size:").pack(side="left", padx=(0, 5))
        self.entry_batch = ctk.CTkEntry(param_frame, width=60)
        self.entry_batch.insert(0, "16")
        self.entry_batch.pack(side="left")
        
        # 2. MENU NÚT CHỨC NĂNG CĂN GIỮA
        self.btn_start_train = ctk.CTkButton(self.train_frame, text="Bắt đầu Huấn luyện YOLOv8", fg_color="#1f538d", hover_color="#14375d", font=("Arial", 14, "bold"), height=40, command=self.start_training)
        self.btn_start_train.pack(pady=15)
        
        # 3. KHUNG HIỂN THỊ LOG TIẾN ĐỘ
        self.log_group = ctk.CTkFrame(self.train_frame, border_width=1, border_color="#555555", fg_color="#1e1e1e")
        self.log_group.pack(fill="both", expand=True, pady=5)
        
        ctk.CTkLabel(self.log_group, text=" Terminal Output (Log) ", text_color="#aaaaaa").place(x=10, y=-12)
        
        self.train_log_textbox = ctk.CTkTextbox(self.log_group, fg_color="#000000", text_color="#00ff00", font=("Consolas", 12))
        self.train_log_textbox.pack(fill="both", expand=True, padx=10, pady=(15, 10))
        self.train_log_textbox.insert("0.0", "--- Sẵn sàng Huấn luyện YOLOv8 ---\n\n")
        self.train_log_textbox.configure(state="disabled")

    def browse_yaml_file(self):
        file_path = filedialog.askopenfilename(title="Chọn file data.yaml", filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")])
        if file_path:
            self.entry_yaml_path.configure(state="normal")
            self.entry_yaml_path.delete(0, 'end')
            self.entry_yaml_path.insert(0, file_path)
            self.entry_yaml_path.configure(state="readonly")
            
    def start_training(self):
        yaml_path = self.entry_yaml_path.get()
        if not yaml_path:
            self.append_log("Lỗi: Vui lòng chọn file cấu hình (data.yaml) của Dataset.")
            return
            
        epochs = self.entry_epochs.get()
        batch_size = self.entry_batch.get()
        
        self.append_log(f"Đang chuẩn bị khởi động YOLOv8 Trainer...")
        self.append_log(f"Dataset: {yaml_path}")
        self.append_log(f"Epochs: {epochs} | Batch size: {batch_size}")
        
        self.btn_start_train.configure(state="disabled", text="Đang Huấn Luyện...")
        
        # Khởi chạy Trainer ngầm
        try:
            epochs_int = int(epochs)
            batch_int = int(batch_size)
            from models.trainer import YOLOTrainer
            self.trainer = YOLOTrainer(
                yaml_path=yaml_path,
                epochs=epochs_int,
                batch_size=batch_int,
                log_callback=self.append_log,
                finish_callback=self.on_training_finished
            )
            self.trainer.start()
        except Exception as e:
            self.append_log(f"Lỗi khởi tạo huấn luyện: {e}")
            self.on_training_finished()

    def on_training_finished(self):
        # Trả lại trạng thái nút sau khi Train xong hoặc Lỗi
        self.btn_start_train.configure(state="normal", text="Bắt đầu Huấn luyện YOLOv8")

    def append_log(self, text):
        self.train_log_textbox.configure(state="normal")
        self.train_log_textbox.insert("end", text + "\n")
        self.train_log_textbox.see("end")
        self.train_log_textbox.configure(state="disabled")

    def on_tree_select(self, event):
        """ Xử lý sự kiện khi người dùng click vào một nốt trong bảng kết quả """
        # Lấy item đang được click
        selected_items = self.tree.selection()
        if not selected_items: return
            
        # Lấy dữ liệu của dòng đó (values trả về tuple)
        item_values = self.tree.item(selected_items[0])["values"]
        
        if item_values:
            # Vị trí lát cắt CT nằm ở cột Z (Index 2 của bảng ID, Voxel, Z, Y, X)
            z_slice = int(item_values[2])
            
            # Nếu lát cắt hợp lệ
            if 0 <= z_slice < len(self.ct_images):
                # Chuyển Slider UI về giá trị đó
                self.slice_slider.set(z_slice)
                # Tự động gọi thủ công hàm render slice
                self.on_slider_change(z_slice)

    def browse_dicom_folder(self):
        dir_path = filedialog.askdirectory(title="Chọn thư mục chứa DICOM Series")
        if not dir_path: return
            
        self.entry_dicom_path.configure(state="normal")
        self.entry_dicom_path.delete(0, 'end')
        self.entry_dicom_path.insert(0, dir_path)
        self.entry_dicom_path.configure(state="readonly")
        
        self.lbl_status.configure(text=f"Đang đọc thư mục: {dir_path}", text_color="yellow")
        self.update_idletasks()
        
        try:
            pil_images = load_dicom_series(dir_path, target_size=(400, 400))
            if pil_images and len(pil_images) > 0:
                self.ct_images = pil_images
                self.current_slice_index = 0
                
                self.slice_slider.configure(state="normal", from_=0, to=len(self.ct_images) - 1, number_of_steps=len(self.ct_images) - 1)
                self.slice_slider.set(0)
                self.display_slice()
                
                self.lbl_status.configure(text=f"Đã tải {len(self.ct_images)} lát cắt. Phát hiện 0 nốt.", text_color="white")
            else:
                self.lbl_status.configure(text="Lỗi: Thư mục vừa chọn không có ảnh DICOM hợp lệ!", text_color="red")
        except Exception as e:
            self.lbl_status.configure(text=f"Lỗi đọc DICOM: {e}", text_color="red")
            
    def browse_model_file(self):
        file_path = filedialog.askopenfilename(title="Chọn file mô hình YOLO (.pt)", filetypes=[("PyTorch Weights", "*.pt"), ("All Files", "*.*")])
        if file_path:
            self.entry_model_path.configure(state="normal")
            self.entry_model_path.delete(0, 'end')
            self.entry_model_path.insert(0, os.path.basename(file_path))
            self.entry_model_path.configure(state="readonly")
            
            try:
                self.lbl_status.configure(text=f"Đang tải mô hình YOLOv8 mới...", text_color="yellow")
                self.update_idletasks()
                
                self.ai_pipeline.load_yolo_weights(file_path)
                
                self.lbl_status.configure(text=f"Đã tải thành công mô hình: {os.path.basename(file_path)}!", text_color="#00ff00")
            except Exception as e:
                self.lbl_status.configure(text=f"Lỗi tải trọng số YOLOv8: {e}", text_color="red")

    def on_slider_change(self, value):
        if self.ct_images:
            self.current_slice_index = int(value)
            self.display_slice()

    def display_slice(self):
        if not self.ct_images: return
        
        # Lấy ảnh PIL gốc
        pil_image_orig = self.ct_images[self.current_slice_index]
        
        # Nếu có phân tích, chuyển đổi sang RGB Numpy để vẽ mask mờ lên
        if hasattr(self, 'analysis_results') and self.current_slice_index in self.analysis_results:
            result = self.analysis_results[self.current_slice_index]
            
            # Chuyển L (Grayscale 0-255) thành RGB để vẽ màu đỏ
            img_rgb_np = np.array(pil_image_orig.convert("RGB"))
            
            for nodule in result.get("nodules", []):
                x1, y1, x2, y2 = nodule["x1"], nodule["y1"], nodule["x2"], nodule["y2"]
                fine_mask = nodule.get("fine_mask", None)
                
                if fine_mask is not None:
                    # Tạo màu overlay đỏ (R, G, B) = (255, 0, 0)
                    color = np.array([255, 0, 0], dtype=np.uint8)
                    
                    # Lấy vùng ảnh gốc tương ứng với BBox
                    roi = img_rgb_np[y1:y2, x1:x2]
                    
                    # Alpha blending (Kênh làm mờ)
                    alpha = 0.5
                    
                    # Chỉ tô đỏ chỗ nào fine_mask = True
                    for c in range(3): # RGB
                        roi[:, :, c] = np.where(fine_mask, 
                                               (alpha * color[c] + (1 - alpha) * roi[:, :, c]).astype(np.uint8), 
                                               roi[:, :, c])
                                               
                    # Cập nhật roi mới vào hình gốc
                    img_rgb_np[y1:y2, x1:x2] = roi

            # Chuyển ngược về PIL Image sau khi đã tô màu
            pil_image = Image.fromarray(img_rgb_np)
        else:
            pil_image = pil_image_orig.copy()
                
        # Hiển thị lên CustomTkinter
        ctk_img = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(400, 400))
        self.image_label.configure(image=ctk_img, text="")
        self.lbl_slice.configure(text=f"Lát cắt: {self.current_slice_index}")

    def run_analysis(self):
        if not self.ct_images:
            self.lbl_status.configure(text="Vui lòng tải thư mục DICOM trước.", text_color="orange")
            return
            
        self.lbl_status.configure(text="Hệ thống đang phân tích ảnh CT bằng AI...", text_color="yellow")
        self.update_idletasks()
        
        # Xóa các biến kết quả cũ đi
        self.analysis_results = {}
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # ----------------------------------------------------
        # Chạy AI Pipeline cho toàn bộ lát cắt
        # ----------------------------------------------------
        total_nodules = 0
        nodule_id_counter = 1
        
        for i, pil_image in enumerate(self.ct_images):
            # Cập nhật UI để cho thấy tiến độ (tránh đơ app quá lâu)
            if i % 5 == 0:
                self.lbl_status.configure(text=f"Đang xử lý lớp cắt {i+1}/{len(self.ct_images)}...")
                self.update_idletasks()
                
            # Chạy qua Pipeline AI
            result = self.ai_pipeline.run_full_pipeline(pil_image)
            self.analysis_results[i] = result
            
            # Cập nhật thông số
            nodules_in_slice = result.get("nodules", [])
            total_nodules += len(nodules_in_slice)
            
            # Chèn dòng vào Cột Kết Quả (Treeview)
            for n in nodules_in_slice:
                conf = f"{n['confidence']*100:.1f}%"
                # ID, Voxel, Z, Y, X, Malignancy, Color
                self.tree.insert("", "end", values=(nodule_id_counter, n['voxel'], i, n['center_y'], n['center_x'], conf, "Red"))
                nodule_id_counter += 1
                
        # Cập nhật lại khung hiển thị hiện tại để vẽ BBox nếu lát cắt đang xem có nốt
        self.display_slice()
            
        self.lbl_status.configure(text=f"Phân tích hoàn tất: Phát hiện {total_nodules} nốt nghi ngờ.", text_color="white")

if __name__ == "__main__":
    app = LungNoduleApp()
    app.mainloop()
