import customtkinter as ctk
from tkinter import filedialog, ttk
from PIL import Image
import os
import numpy as np
import threading
from utils.image_reader import load_dicom_series
from pipeline import AIPipeline
from ui.settings_panel import SettingsPanel
from ui.result_tree import ResultTree
from utils.cluster_3d import cluster_nodules_3d

class AnalysisTab(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.ct_images = []
        self.current_slice_index = 0
        self.analysis_results = {}
        self.ai_pipeline = AIPipeline(device='cpu')
        self.setup_ui()

    def set_status(self, text, color="white"):
        self.lbl_status.configure(text=text, text_color=color)
        self.update_idletasks()

    def setup_ui(self):
        # 0. KHUNG TRÊN CÙNG: Chỉ có 1 cột
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.pack(fill="x", pady=(5, 5))
        top_frame.columnconfigure(0, weight=1)

        # 1. NGUỒN DỮ LIỆU
        data_grp = ctk.CTkFrame(top_frame, border_width=1, border_color="#555555", fg_color="#2b2b2b")
        data_grp.grid(row=0, column=0, sticky="nsew", padx=(0, 0), ipadx=5, ipady=5)
        
        header_frame = ctk.CTkFrame(data_grp, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(5, 0))
        ctk.CTkLabel(header_frame, text="Nguồn dữ liệu", text_color="#aaaaaa", font=("Segoe UI", 12)).pack(side="left")
        
        self.settings = SettingsPanel(self)
        ctk.CTkButton(header_frame, text="⚙ Cài đặt cấu hình AI", width=120, fg_color="#444", hover_color="#555", 
                      command=self.settings.show_window).pack(side="right")
        data_content = ctk.CTkFrame(data_grp, fg_color="transparent")
        data_content.pack(fill="x", padx=10, pady=(2, 5))
        data_content.columnconfigure(1, weight=1)
        ctk.CTkLabel(data_content, text="DICOM Dir").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry_dicom = ctk.CTkEntry(data_content, state="readonly")
        self.entry_dicom.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(data_content, text="Chọn Thư Mục...", width=100, command=self.browse_dicom).grid(row=0, column=2, padx=5, pady=5)
        ctk.CTkLabel(data_content, text="Model (.pt)").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entry_model = ctk.CTkEntry(data_content)
        self.entry_model.insert(0, "yolov8n.pt")
        self.entry_model.configure(state="readonly")
        self.entry_model.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(data_content, text="Đổi Model AI...", width=100, command=self.browse_model).grid(row=1, column=2, padx=5, pady=5)

        # 2. KHÔNG CÒN KHUNG THIẾT LẬP (Đã chuyển thành Cửa sổ riêng SettingsPanel)

        # 3. NÚT PHÂN TÍCH
        ctk.CTkButton(self, text="Phân tích", fg_color="#3b3b3b", hover_color="#4b4b4b", command=self.run_analysis).pack(fill="x", pady=10)
        self.lbl_status = ctk.CTkLabel(self, text="Phát hiện 0 nốt.", anchor="w")
        self.lbl_status.pack(fill="x", pady=(0, 5))

        # 3. KẾT QUẢ VÀ HÌNH ẢNH
        bot_frame = ctk.CTkFrame(self, fg_color="transparent")
        bot_frame.pack(fill="both", expand=True)
        bot_frame.columnconfigure((0,1), weight=1)
        bot_frame.rowconfigure(0, weight=1)
        # 3.1 BẢNG TREEVIEW
        self.result_tree = ResultTree(bot_frame, on_item_click_cb=self.go_to_slice)
        self.result_tree.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        # 3.2 HÌNH ẢNH
        img_frame = ctk.CTkFrame(bot_frame, fg_color="#2b2b2b", border_width=1, border_color="#555555")
        img_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        sld_frame = ctk.CTkFrame(img_frame, fg_color="transparent")
        sld_frame.pack(side="bottom", fill="x", padx=10, pady=(0, 10))
        self.lbl_slice = ctk.CTkLabel(sld_frame, text="Lát cắt: 0", width=80)
        self.lbl_slice.pack(side="left", padx=5)
        
        self.slider = ctk.CTkSlider(sld_frame, from_=0, to=1, command=self.on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=5)
        self.slider.set(0); self.slider.configure(state="disabled")

        self.img_lbl = ctk.CTkLabel(img_frame, text="")
        self.img_lbl.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Lắng nghe sự kiện thay đổi kích thước khung ảnh để Responsive
        img_frame.bind("<Configure>", self._on_img_frame_resize)

    def browse_dicom(self):
        d_path = filedialog.askdirectory(title="Chọn thư mục chứa DICOM")
        if not d_path: return
        self.entry_dicom.configure(state="normal")
        self.entry_dicom.delete(0, 'end'); self.entry_dicom.insert(0, d_path)
        self.entry_dicom.configure(state="readonly")
        
        self.set_status(f"Đang đọc: {d_path}", "yellow")
        
        def _load_thread():
            try:
                imgs = load_dicom_series(d_path, target_size=(512, 512))
                self.after(0, self._on_dicom_loaded, imgs)
            except Exception as e:
                self.after(0, self.set_status, f"Lỗi: {e}", "red")
                
        threading.Thread(target=_load_thread, daemon=True).start()

    def _on_dicom_loaded(self, imgs):
        if imgs:
            self.ct_images = imgs
            self.slider.configure(state="normal", from_=0, to=len(imgs)-1, number_of_steps=len(imgs)-1)
            self.slider.set(0); self.current_slice_index = 0
            self.display_slice(); self.set_status(f"Đã tải {len(imgs)} lát cắt.", "white")
        else:
            self.set_status("Lỗi: Không có ảnh DICOM!", "red")

    def browse_model(self):
        f_path = filedialog.askopenfilename(title="Chọn Model YOLO (.pt)", filetypes=[("PyTorch", "*.pt")])
        if not f_path: return
        self.entry_model.configure(state="normal")
        self.entry_model.delete(0, 'end'); self.entry_model.insert(0, os.path.basename(f_path))
        self.entry_model.configure(state="readonly")
        
        def _load_model_thread():
            try:
                self.after(0, self.set_status, "Đang tải YOLOv8 mới...", "yellow")
                self.ai_pipeline.load_yolo_weights(f_path)
                self.after(0, self.set_status, f"Đã tải {os.path.basename(f_path)}!", "#00ff00")
            except Exception as e:
                self.after(0, self.set_status, f"Lỗi tải YOLO: {e}", "red")

        threading.Thread(target=_load_model_thread, daemon=True).start()

    def on_slider(self, val):
        if self.ct_images: self.current_slice_index = int(val); self.display_slice()

    def go_to_slice(self, z_idx):
        if 0 <= z_idx < len(self.ct_images): self.slider.set(z_idx); self.on_slider(z_idx)

    def run_analysis(self):
        if not self.ct_images:
            self.set_status("Vui lòng tải DICOM trước.", "orange")
            return
        self.set_status("Hệ thống đang phân tích AI...", "yellow")
        self.analysis_results = {}
        self.result_tree.clear()
        
        self.slider.configure(state="disabled")
        
        conf = self.settings.get_conf_threshold()
        voxel_min = self.settings.get_min_voxel()
        fpr_thresh = self.settings.get_fpr_threshold()
        min_slices = self.settings.get_min_slices()

        def _analysis_thread():
            try:
                total_slices = 0
                for i, img in enumerate(self.ct_images):
                    if i % 10 == 0: 
                        self.after(0, self.set_status, f"Đang quét từng lát 2D: {i+1}/{len(self.ct_images)}...")
                    
                    res = self.ai_pipeline.run_full_pipeline(
                        img, full_volume=self.ct_images, slice_idx=i,
                        conf_threshold=conf, min_voxel=voxel_min, fpr_threshold=fpr_thresh
                    )
                    
                    self.analysis_results[i] = res
                    total_slices += 1

                self.after(0, self.set_status, "Đang nhào nặn Không gian 3D (Clustering)...")
                
                clusters = cluster_nodules_3d(self.analysis_results, min_slices=min_slices)
                
                self.after(0, self._on_analysis_done, clusters)
            except Exception as e:
                self.after(0, self.set_status, f"Lỗi phân tích: {e}", "red")

        threading.Thread(target=_analysis_thread, daemon=True).start()

    def _on_analysis_done(self, clusters):
        n_id = 1
        for c in clusters:
            fpr_percent = f"{c.get('fpr_score', 1.0)*100:.1f}% AI3D" 
            
            # Hiện Range "Lát 22-25" nếu nốt kéo dài, ngược lại hiện "Lát 22"
            z_val = f"{c['z_start']}-{c['z_end']}" if c['z_start'] != c['z_end'] else str(c['z_start'])
            
            self.result_tree.add_item((n_id, c['voxel'], z_val, c['center_y'], c['center_x'], fpr_percent, "Red"))
            n_id += 1
        
        self.display_slice()
        self.slider.configure(state="normal")
        self.set_status(f"Hoàn tất: Đã gộp thành {len(clusters)} Khối U 3D.", "white")

    def _get_display_size(self):
        """Tính kích thước ảnh vuông lớn nhất vừa vặn trong khung chứa."""
        w = self.img_lbl.winfo_width()
        h = self.img_lbl.winfo_height()
        if w < 50 or h < 50:
            return 350
        size = max(200, min(w, h) - 60)
        return min(size, 550)

    def _on_img_frame_resize(self, event=None):
        """Callback khi khung ảnh thay đổi kích thước -> vẽ lại ảnh."""
        if self.ct_images:
            self.display_slice()

    def display_slice(self):
        if not self.ct_images: return
        img_orig = self.ct_images[self.current_slice_index]
        if self.current_slice_index in self.analysis_results:
            res = self.analysis_results[self.current_slice_index]
            np_img = np.array(img_orig.convert("RGB"))
            for n in res.get("nodules", []):
                x1, y1, x2, y2 = n["x1"], n["y1"], n["x2"], n["y2"]
                if n.get("fine_mask") is not None:
                    roi = np_img[y1:y2, x1:x2]
                    for c in range(3):
                        roi[:,:,c] = np.where(n["fine_mask"], (0.5 * np.array([255,0,0])[c] + 0.5 * roi[:,:,c]).astype(np.uint8), roi[:,:,c])
                    np_img[y1:y2, x1:x2] = roi
            img_final = Image.fromarray(np_img)
        else: img_final = img_orig.copy()
        
        # Tự động co giãn ảnh theo kích thước cửa sổ
        display_size = self._get_display_size()
        self.img_lbl.configure(image=ctk.CTkImage(light_image=img_final, dark_image=img_final, size=(display_size, display_size)), text="")
        self.lbl_slice.configure(text=f"Lát cắt: {self.current_slice_index}")
