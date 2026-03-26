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
from ui.image_viewer import ImageViewer
from utils.cluster_3d import cluster_nodules_3d

class AnalysisTab(ctk.CTkFrame):
    def __init__(self, master, on_images_loaded=None, on_results_ready=None, on_clusters_ready=None, on_fill_changed=None, **kwargs):
        super().__init__(master, **kwargs)
        self.on_images_loaded = on_images_loaded
        self.on_results_ready = on_results_ready
        self.on_clusters_ready = on_clusters_ready
        self.on_fill_changed = on_fill_changed
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
        
        self.settings = SettingsPanel(self, on_model_changed=self._load_yolo_model)
        ctk.CTkButton(header_frame, text="⚙ Cài đặt cấu hình AI", width=120, fg_color="#444", hover_color="#555", 
                      command=self.settings.show_window).pack(side="right")
        data_content = ctk.CTkFrame(data_grp, fg_color="transparent")
        data_content.pack(fill="x", padx=10, pady=(2, 5))
        data_content.columnconfigure(1, weight=1)
        ctk.CTkLabel(data_content, text="DICOM Dir").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry_dicom = ctk.CTkEntry(data_content, state="readonly")
        self.entry_dicom.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(data_content, text="Chọn Thư Mục...", width=100, command=self.browse_dicom).grid(row=0, column=2, padx=5, pady=5)

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
        self.image_viewer = ImageViewer(bot_frame, self.settings)
        self.image_viewer.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

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
            self.image_viewer.set_images(imgs)
            if self.on_images_loaded:
                self.on_images_loaded(imgs)
            self.set_status(f"Đã tải {len(imgs)} lát cắt.", "white")
        else:
            self.set_status("Lỗi: Không có ảnh DICOM!", "red")

    def _load_yolo_model(self, f_path):
        import os
        def _load_model_thread():
            try:
                self.after(0, self.set_status, "Đang tải YOLOv8 mới...", "yellow")
                self.ai_pipeline.load_yolo_weights(f_path)
                self.after(0, self.set_status, f"Đã tải {os.path.basename(f_path)}!", "#00ff00")
            except Exception as e:
                self.after(0, self.set_status, f"Lỗi tải YOLO: {e}", "red")

        threading.Thread(target=_load_model_thread, daemon=True).start()

    def go_to_slice(self, z_idx):
        self.image_viewer.go_to_slice(z_idx)
        # Chuyển focus ra khỏi Treeview để tránh kẹt phím mũi tên
        self.focus_set()

    def step_slice(self, delta):
        imgs = self.image_viewer.get_images()
        if not imgs:
            return
        cur = self.image_viewer.current_slice_index
        max_idx = len(imgs) - 1
        next_idx = max(0, min(cur + int(delta), max_idx))
        if next_idx != cur:
            self.image_viewer.go_to_slice(next_idx)

    def run_analysis(self):
        imgs = self.image_viewer.get_images()
        if not imgs:
            self.set_status("Vui lòng tải DICOM trước.", "orange")
            return
        if self.on_fill_changed:
            self.on_fill_changed(self.settings.get_fill_color_enabled())
        self.set_status("Hệ thống đang phân tích AI...", "yellow")
        self.image_viewer.analysis_results = {}
        self.result_tree.clear()
        
        conf = self.settings.get_conf_threshold()
        voxel_min = self.settings.get_min_voxel()
        fpr_thresh = self.settings.get_fpr_threshold()
        min_slices = self.settings.get_min_slices()

        def _analysis_thread():
            try:
                total_slices = 0
                for i, img in enumerate(imgs):
                    if i % 10 == 0: 
                        self.after(0, self.set_status, f"Đang quét từng lát 2D: {i+1}/{len(imgs)}...")
                    
                    res = self.ai_pipeline.run_full_pipeline(
                        img, full_volume=imgs, slice_idx=i,
                        conf_threshold=conf, min_voxel=voxel_min, fpr_threshold=fpr_thresh
                    )
                    
                    self.image_viewer.update_analysis_result(i, res)
                    total_slices += 1

                self.after(0, self.set_status, "Đang nhào nặn Không gian 3D (Clustering)...")
                
                clusters = cluster_nodules_3d(self.image_viewer.analysis_results, min_slices=min_slices)
                
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
        
        self.image_viewer.display_slice()
        if self.on_results_ready:
            self.on_results_ready(self.image_viewer.analysis_results)
        if self.on_clusters_ready:
            self.on_clusters_ready(clusters)
        self.set_status(f"Hoàn tất: Đã gộp thành {len(clusters)} Khối U 3D.", "white")
