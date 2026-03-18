import customtkinter as ctk
import json
import os

class SettingsPanel(ctk.CTkToplevel):
    """Cửa sổ Thiết lập tham số AI - cho phép người dùng tùy chỉnh các ngưỡng lọc."""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.title("Cài đặt cấu hình AI")
        self.geometry("400x250")
        self.resizable(False, False)
        
        self.settings_file = "settings.json"
        
        # Ẩn cửa sổ thay vì tắt khi nhấn X
        self.protocol("WM_DELETE_WINDOW", self.hide_window)
        self.withdraw()
        
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)
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

        # 4. Số lát cắt tối thiểu (Min Slices)
        ctk.CTkLabel(content, text="Số lát cắt tối thiểu").grid(row=3, column=0, padx=5, pady=6, sticky="w")
        self.entry_slices = ctk.CTkEntry(content, width=60)
        self.entry_slices.grid(row=3, column=1, padx=5, pady=6, sticky="w")
        
        self.load_settings()

    def load_settings(self):
        default_settings = {"conf": 0.25, "voxel": 50, "fpr": 0.50, "slices": 3}
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                    default_settings.update(settings)
            except: pass
            
        self.slider_conf.set(default_settings["conf"])
        self.lbl_conf.configure(text=f'{default_settings["conf"]:.2f}')
        
        self.entry_voxel.delete(0, 'end')
        self.entry_voxel.insert(0, str(default_settings["voxel"]))
        
        self.slider_fpr.set(default_settings["fpr"])
        self.lbl_fpr.configure(text=f'{default_settings["fpr"]:.2f}')
        
        self.entry_slices.delete(0, 'end')
        self.entry_slices.insert(0, str(default_settings["slices"]))

    def save_settings(self):
        settings = {
            "conf": self.get_conf_threshold(),
            "voxel": self.get_min_voxel(),
            "fpr": self.get_fpr_threshold(),
            "slices": self.get_min_slices()
        }
        try:
            with open(self.settings_file, "w") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Lỗi lưu cài đặt: {e}")

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

    def get_min_slices(self):
        try: return int(self.entry_slices.get())
        except: return 3

    def show_window(self):
        self.deiconify()
        self.focus()

    def hide_window(self):
        self.save_settings()
        self.withdraw()
