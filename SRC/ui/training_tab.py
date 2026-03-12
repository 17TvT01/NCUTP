import customtkinter as ctk
from tkinter import filedialog
from models.trainer import YOLOTrainer

class TrainingTab(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.setup_ui()

    def setup_ui(self):
        # 1. KHU VỰC CẤU HÌNH DATASET
        self.train_config_group = ctk.CTkFrame(self, border_width=1, border_color="#555555", fg_color="#2b2b2b")
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
        self.btn_start_train = ctk.CTkButton(self, text="Bắt đầu Huấn luyện YOLOv8", fg_color="#1f538d", hover_color="#14375d", font=("Arial", 14, "bold"), height=40, command=self.start_training)
        self.btn_start_train.pack(pady=15)
        
        # 3. KHUNG HIỂN THỊ LOG TIẾN ĐỘ
        self.log_group = ctk.CTkFrame(self, border_width=1, border_color="#555555", fg_color="#1e1e1e")
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
        self.btn_start_train.configure(state="normal", text="Bắt đầu Huấn luyện YOLOv8")

    def append_log(self, text):
        self.train_log_textbox.configure(state="normal")
        self.train_log_textbox.insert("end", text + "\n")
        self.train_log_textbox.see("end")
        self.train_log_textbox.configure(state="disabled")
