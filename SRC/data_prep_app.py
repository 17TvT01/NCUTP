import customtkinter as ctk
from tkinter import filedialog, END
import threading
import os
import sys

# Đảm bảo có thể import từ src.utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.data_prep import create_dataset

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class RedirectStdout:
    """Class hỗ trợ chuyển hướng print() từ console vào textbox"""
    def __init__(self, textbox):
        self.textbox = textbox

    def write(self, string):
        self.textbox.insert(END, string)
        self.textbox.see(END)

    def flush(self):
        pass

class DataPrepApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Trình Sinh Dữ liệu Huấn luyện (Data Augmentation) - NCS")
        self.geometry("800x650")
        
        # Danh sách các folder
        self.folders = []
        
        # UI Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        
        # -- KHUNG 1: Cấu hình chung --
        frame_config = ctk.CTkFrame(self)
        frame_config.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(frame_config, text="Thư mục Output:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.entry_out = ctk.CTkEntry(frame_config, width=300)
        self.entry_out.insert(0, "dataset_yolo_final")
        self.entry_out.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        ctk.CTkButton(frame_config, text="Chọn Chỗ Lưu...", command=self._browse_out, width=120).grid(row=0, column=2, padx=10, pady=10)
        
        ctk.CTkLabel(frame_config, text="Hệ số Augment (x):").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.entry_aug = ctk.CTkEntry(frame_config, width=100)
        self.entry_aug.insert(0, "5")
        self.entry_aug.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        # -- KHUNG 2: Quản lý thư mục --
        frame_folders = ctk.CTkFrame(self)
        frame_folders.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(frame_folders, text="Danh sách Thư mục DICOM + XML (10 folders mẫu):").pack(anchor="w", padx=10, pady=5)
        
        self.listbox = ctk.CTkTextbox(frame_folders, height=150)
        self.listbox.pack(fill="x", padx=10, pady=5)
        self.listbox.configure(state="disabled")
        
        btn_frame = ctk.CTkFrame(frame_folders, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(btn_frame, text="Nhập Thư mục...", command=self._add_folder).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Xóa sạch", command=self._clear_folders, fg_color="tomato", hover_color="darkred").pack(side="left", padx=5)
        
        # Nút Chạy to nhất
        self.btn_run = ctk.CTkButton(frame_folders, text="BẮT ĐẦU SINH DỮ LIỆU", command=self._start_process, 
                                     height=40, font=("Arial", 14, "bold"), fg_color="green", hover_color="darkgreen")
        self.btn_run.pack(side="right", padx=5)
        
        # -- KHUNG 3: Logs --
        frame_logs = ctk.CTkFrame(self)
        frame_logs.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(frame_logs, text="Tiến trình xử lý (Logs):").pack(anchor="w", padx=10, pady=5)
        self.log_box = ctk.CTkTextbox(frame_logs, font=("Consolas", 12))
        self.log_box.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Đổi luồng print vào log box
        sys.stdout = RedirectStdout(self.log_box)
        sys.stderr = RedirectStdout(self.log_box)

    def _browse_out(self):
        d = filedialog.askdirectory(title="Chọn thư mục xuất Dataset")
        if d:
            self.entry_out.delete(0, 'end')
            self.entry_out.insert(0, d)
            
    def _add_folder(self):
        d = filedialog.askdirectory(title="Chọn thư mục chứa DICOM và XML")
        if d and d not in self.folders:
            self.folders.append(d)
            self._render_listbox()
            
    def _clear_folders(self):
        self.folders = []
        self._render_listbox()
        
    def _render_listbox(self):
        self.listbox.configure(state="normal")
        self.listbox.delete("1.0", END)
        for i, f in enumerate(self.folders):
            self.listbox.insert(END, f"[{i+1}] {f}\n")
        self.listbox.configure(state="disabled")

    def _start_process(self):
        if not self.folders:
            print("❌ Lỗi: Bạn chưa chọn thư mục DICOM nào!")
            return
            
        out_dir = self.entry_out.get().strip()
        try:
            aug_val = int(self.entry_aug.get().strip())
        except ValueError:
            print("❌ Lỗi: Hệ số Augment phải là số nguyên (như 5, 10)!")
            return
            
        self.btn_run.configure(state="disabled", text="ĐANG XỬ LÝ...")
        self.log_box.delete("1.0", END)
        print(f"BẮT ĐẦU CHUỖI XỬ LÝ {len(self.folders)} THƯ MỤC...\n")
        
        # Chạy thread dưới nền để chống đơ UI
        threading.Thread(target=self._run_batch, args=(out_dir, aug_val), daemon=True).start()

    def _run_batch(self, out_dir, aug_val):
        try:
            for idx, folder in enumerate(self.folders):
                print(f"={'='*50}")
                print(f"🚀 XỬ LÝ THƯ MỤC {idx+1}/{len(self.folders)}: {folder}")
                print(f"={'='*50}")
                
                # Hàm create_dataset đọc và ghi thêm vào out_dir
                create_dataset(
                    image_dir=folder, 
                    xml_dir=folder, 
                    output_dir=out_dir, 
                    classes=["nodule"], 
                    augment_factor=aug_val
                )
            
            print("\n🎉 HOÀN TẤT TẤT CẢ QUÁ TRÌNH SINH DỮ LIỆU!")
            print(f"Toàn bộ dữ liệu của {len(self.folders)} mục đã được dồn vào: {os.path.abspath(out_dir)}")
            print("Bây giờ bạn có thể mang bộ dữ liệu này ném vào phần Huấn Luyện AI!")
            
        except Exception as e:
            print(f"\n❌ LÕI CRITICAL TRONG QUÁ TRÌNH XỬ LÝ: {str(e)}")
            
        finally:
            self.btn_run.configure(state="normal", text="BẮT ĐẦU SINH DỮ LIỆU")

if __name__ == "__main__":
    app = DataPrepApp()
    app.mainloop()
