import customtkinter as ctk

class SettingsPanel(ctk.CTkFrame):
    """Khung Thiết lập tham số AI - cho phép người dùng tùy chỉnh các ngưỡng lọc."""
    def __init__(self, master, **kwargs):
        super().__init__(master, border_width=1, border_color="#555555", fg_color="#2b2b2b", **kwargs)
        ctk.CTkLabel(self, text="Thiết lập", text_color="#aaaaaa", font=("Segoe UI", 12)).pack(anchor="w", padx=10, pady=(5, 0))
        
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=10, pady=(2, 10))
        content.columnconfigure(1, weight=1)

        # 1. Ngưỡng xác suất YOLO (Confidence Threshold)
        ctk.CTkLabel(content, text="Ngưỡng xác suất").grid(row=0, column=0, padx=5, pady=6, sticky="w")
        self.lbl_conf = ctk.CTkLabel(content, text="0.25", width=40)
        self.lbl_conf.grid(row=0, column=2, padx=5, pady=6)
        self.slider_conf = ctk.CTkSlider(content, from_=0.05, to=0.95, number_of_steps=90,
                                          command=self._on_conf_change)
        self.slider_conf.set(0.25)
        self.slider_conf.grid(row=0, column=1, padx=5, pady=6, sticky="ew")

        # 2. Voxel tối thiểu (Minimum Area Filter)
        ctk.CTkLabel(content, text="Voxel tối thiểu").grid(row=1, column=0, padx=5, pady=6, sticky="w")
        self.entry_voxel = ctk.CTkEntry(content, width=60)
        self.entry_voxel.insert(0, "50")
        self.entry_voxel.grid(row=1, column=1, padx=5, pady=6, sticky="w")

        # 3. Ngưỡng FPR 3D (FPR Score Threshold)
        ctk.CTkLabel(content, text="Ngưỡng FPR 3D").grid(row=2, column=0, padx=5, pady=6, sticky="w")
        self.lbl_fpr = ctk.CTkLabel(content, text="0.50", width=40)
        self.lbl_fpr.grid(row=2, column=2, padx=5, pady=6)
        self.slider_fpr = ctk.CTkSlider(content, from_=0.1, to=0.99, number_of_steps=89,
                                         command=self._on_fpr_change)
        self.slider_fpr.set(0.50)
        self.slider_fpr.grid(row=2, column=1, padx=5, pady=6, sticky="ew")

    def _on_conf_change(self, val):
        self.lbl_conf.configure(text=f"{val:.2f}")

    def _on_fpr_change(self, val):
        self.lbl_fpr.configure(text=f"{val:.2f}")

    def get_conf_threshold(self):
        return self.slider_conf.get()

    def get_min_voxel(self):
        try: return int(self.entry_voxel.get())
        except: return 50

    def get_fpr_threshold(self):
        return self.slider_fpr.get()
