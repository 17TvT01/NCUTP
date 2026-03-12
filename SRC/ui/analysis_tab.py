import customtkinter as ctk
from tkinter import filedialog, ttk
from PIL import Image
import os
import numpy as np
from utils.image_reader import load_dicom_series
from pipeline import AIPipeline
from ui.settings_panel import SettingsPanel

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
        # 0. KHUNG TRÊN CÙNG: Chia 2 cột (Nguồn dữ liệu | Thiết lập)
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.pack(fill="x", pady=(5, 5))
        top_frame.columnconfigure(0, weight=3)
        top_frame.columnconfigure(1, weight=1)

        # 1. NGUỒN DỮ LIỆU (Cột trái)
        data_grp = ctk.CTkFrame(top_frame, border_width=1, border_color="#555555", fg_color="#2b2b2b")
        data_grp.grid(row=0, column=0, sticky="nsew", padx=(0, 5), ipadx=5, ipady=5)
        ctk.CTkLabel(data_grp, text="Nguồn dữ liệu", text_color="#aaaaaa", font=("Segoe UI", 12)).pack(anchor="w", padx=10, pady=(5, 0))
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

        # 2. THIẾT LẬP (Cột phải)
        self.settings = SettingsPanel(top_frame)
        self.settings.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

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
        res_grp = ctk.CTkFrame(bot_frame, border_width=1, border_color="#555555", fg_color="#2b2b2b")
        res_grp.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        ctk.CTkLabel(res_grp, text="Kết quả", text_color="#aaaaaa", font=("Segoe UI", 12)).pack(anchor="w", padx=10, pady=(5, 0))

        res_content = ctk.CTkFrame(res_grp, fg_color="transparent")
        res_content.pack(fill="both", expand=True, padx=10, pady=(2, 10))
        
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#2b2b2b", foreground="#cccccc", fieldbackground="#2b2b2b", borderwidth=0)
        style.configure("Treeview.Heading", background="#333333", foreground="white", borderwidth=0)
        style.map("Treeview", background=[("selected", "#1f538d")])

        self.tree = ttk.Treeview(res_content, columns=("ID", "Voxel", "Z", "Y", "X", "Malig", "Color"), show="headings", height=15)
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=50, anchor="center")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)

        # 3.2 HÌNH ẢNH
        img_frame = ctk.CTkFrame(bot_frame, fg_color="#2b2b2b", border_width=1, border_color="#555555")
        img_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        self.img_lbl = ctk.CTkLabel(img_frame, text="")
        self.img_lbl.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Lắng nghe sự kiện thay đổi kích thước khung ảnh để Responsive
        img_frame.bind("<Configure>", self._on_img_frame_resize)
        
        sld_frame = ctk.CTkFrame(img_frame, fg_color="transparent")
        sld_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.lbl_slice = ctk.CTkLabel(sld_frame, text="Lát cắt: 0", width=80)
        self.lbl_slice.pack(side="left", padx=5)
        
        self.slider = ctk.CTkSlider(sld_frame, from_=0, to=1, command=self.on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=5)
        self.slider.set(0)
        self.slider.configure(state="disabled")

    def browse_dicom(self):
        d_path = filedialog.askdirectory(title="Chọn thư mục chứa DICOM")
        if not d_path: return
        self.entry_dicom.configure(state="normal")
        self.entry_dicom.delete(0, 'end'); self.entry_dicom.insert(0, d_path)
        self.entry_dicom.configure(state="readonly")
        
        self.set_status(f"Đang đọc: {d_path}", "yellow")
        try:
            imgs = load_dicom_series(d_path, target_size=(512, 512))
            if imgs:
                self.ct_images = imgs
                self.slider.configure(state="normal", from_=0, to=len(imgs)-1, number_of_steps=len(imgs)-1)
                self.slider.set(0); self.current_slice_index = 0
                self.display_slice()
                self.set_status(f"Đã tải {len(imgs)} lát cắt.", "white")
            else: self.set_status("Lỗi: Không có ảnh DICOM!", "red")
        except Exception as e: self.set_status(f"Lỗi: {e}", "red")

    def browse_model(self):
        f_path = filedialog.askopenfilename(title="Chọn Model YOLO (.pt)", filetypes=[("PyTorch", "*.pt")])
        if not f_path: return
        self.entry_model.configure(state="normal")
        self.entry_model.delete(0, 'end'); self.entry_model.insert(0, os.path.basename(f_path))
        self.entry_model.configure(state="readonly")
        try:
            self.set_status("Đang tải YOLOv8 mới...", "yellow")
            self.ai_pipeline.load_yolo_weights(f_path)
            self.set_status(f"Đã tải {os.path.basename(f_path)}!", "#00ff00")
        except Exception as e: self.set_status(f"Lỗi tải YOLO: {e}", "red")

    def on_slider(self, val):
        if self.ct_images:
            self.current_slice_index = int(val)
            self.display_slice()

    def on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel: return
        vals = self.tree.item(sel[0])["values"]
        if vals and 0 <= int(vals[2]) < len(self.ct_images):
            self.slider.set(int(vals[2]))
            self.on_slider(int(vals[2]))

    def run_analysis(self):
        if not self.ct_images:
            self.set_status("Vui lòng tải DICOM trước.", "orange")
            return
        self.set_status("Hệ thống đang phân tích AI...", "yellow")
        self.analysis_results = {}
        [self.tree.delete(i) for i in self.tree.get_children()]
        
        total = 0; n_id = 1
        for i, img in enumerate(self.ct_images):
            if i % 5 == 0: self.set_status(f"Đang xử lý {i+1}/{len(self.ct_images)}...")
            # Đọc tham số từ khung Thiết lập
            conf = self.settings.get_conf_threshold()
            voxel_min = self.settings.get_min_voxel()
            fpr_thresh = self.settings.get_fpr_threshold()
            res = self.ai_pipeline.run_full_pipeline(
                img, full_volume=self.ct_images, slice_idx=i,
                conf_threshold=conf, min_voxel=voxel_min, fpr_threshold=fpr_thresh
            )
            
            self.analysis_results[i] = res
            for n in res.get("nodules", []):
                # fpr_score sẽ mang tỷ lệ Cảnh Báo Ung Thư từ Mạng 3D
                fpr_percent = f"{n.get('fpr_score', 1.0)*100:.1f}% AI3D" 
                self.tree.insert("", "end", values=(n_id, n['voxel'], i, n['center_y'], n['center_x'], fpr_percent, "Red"))
                n_id += 1
                total += 1
        
        self.display_slice()
        self.set_status(f"Hoàn tất: Phát hiện {total} nốt.", "white")

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
