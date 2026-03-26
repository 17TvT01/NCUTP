import threading
import sys
import io
import time
import os
from ultralytics import YOLO

class YOLOTrainer:
    def __init__(self, yaml_path, epochs, batch_size, log_callback, finish_callback):
        self.yaml_path = yaml_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_callback = log_callback
        self.finish_callback = finish_callback
        self.is_training = False

    def start(self):
        if self.is_training:
            return
            
        self.is_training = True
        # Khởi chạy training trên một luồng riêng biệt để không làm đơ giao diện CustomTkinter
        self.train_thread = threading.Thread(target=self._run_training_process, daemon=True)
        self.train_thread.start()

    def _run_training_process(self):
        # 1. Khởi tạo model (Hoặc là model y tế pre-trained, hoặc model gốc siêu nhẹ)
        self.log_callback("⚙️ Đang tải mô hình kiến trúc YOLO11n...")
        try:
            model = YOLO("yolo11n.pt") 
            
            self.log_callback(f"🚀 Bắt đầu quá trình Huấn luyện với Epochs={self.epochs}, Batch={self.batch_size}")
            self.log_callback("Vui lòng chờ... (Sẽ mất thời gian tùy thuộc vào cấu hình máy)\n")
            
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.log_callback(f"💻 Đang sử dụng thiết bị xử lý: {device.upper()}")
            
            # 2. Bắt đầu Train
            # Do thư viện Ultralytics in log trực tiếp ra console, ta khó có thể bắt time real-time từng dòng
            # Việc huấn luyện sẽ phân bổ tự động vào GPU nếu có nhánh CUDA
            results = model.train(
                data=self.yaml_path,
                epochs=self.epochs,
                batch=self.batch_size,
                imgsz=640,          # Kích thước ảnh chuẩn YOLO
                device=device,      # Tự động nhận diện Máy Yếu (CPU) hay Máy Mạnh (CUDA)
                project=os.path.abspath('runs'), # Ép lưu kết quả vào ổ đĩa hiện tại của dự án
                name='train',       # Thư mục con là 'train'
                plots=True,         # Gen các biểu đồ báo cáo
                verbose=False       # Giảm bớt log thừa
            )
            
            self.log_callback("\n✅ Huấn luyện hoàn tất thành công!")
            self.log_callback("File trọng số y tế mới (.pt) đã được lưu tại thư mục: runs/detect/train/weights/best.pt")
            
        except Exception as e:
            self.log_callback(f"\n❌ Lỗi trong quá trình huấn luyện: {str(e)}")
            
        finally:
            self.is_training = False
            if self.finish_callback:
                self.finish_callback()
