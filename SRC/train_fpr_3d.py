import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from models.fpr_3d_net import Lightweight3DCNN

# 1. Định nghĩa chuẩn Dataset Pytorch đọc từ numpy nén
class Rubik3DDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        
        # Lấy X shape (B, 16, 32, 32) -> Chuyển thành Pytorch Type Float32
        self.x = torch.tensor(data['x'], dtype=torch.float32)
        
        # Thêm chiều Channel (C=1) vào -> Output (B, 1, 16, 32, 32)
        if len(self.x.shape) == 4:
            self.x = self.x.unsqueeze(1)
            
        # Pytorch cần chia Normalize cường độ sáng về [0, 1] cho mạng Neural ăn
        self.x = self.x / 255.0  
        
        # Labels 0 (False) và 1 (Nodule)
        self.y = torch.tensor(data['y'], dtype=torch.float32).unsqueeze(1) # Định dạng cột

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def train_3d_classifier(npz_file, epochs=50, batch_size=16, learning_rate=0.001, save_path="weights/fpr_3d_best.pth"):
    # Chọn Thiết bị học (Bắt buộc dùng CUDA nếu muốn nhanh)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔄 Khởi động trình huấn luyện FPR 3D trên: {device.type.upper()}")
    
    # 2. Xây Data Loader
    dataset = Rubik3DDataset(npz_file)
    # Tách 80% Train, 20% Val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # DataLoader Batch Size nhỏ (16-32) để chống tràn VRAM GTX 1650
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Khai báo Mạng "Kiến 3D"
    model = Lightweight3DCNN().to(device)
    
    # 4. Optimizier & Hàm mất mát Nhị Phân (Binary Cross Entropy)
    criterion = nn.BCEWithLogitsLoss() # Thay phiên tính luôn Sigmoid bên trong Loss để tối ưu Gradient
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4) # AdamW có L2 Regularization (Cách điệu rác mọc tua ngoài viền)
    
    # Tính năng MIXED PRECISION (AMP): Vị cứu tinh của RAM 4GB
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    
    best_val_loss = float('inf')
    
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN: Tập Train ({train_size} khối) | Tập Validation ({val_size} khối)")
    print("-" * 50)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # ---- Vòng lặp HỌC CHÍNH ----
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # ÉP FLOAT16 chạy ở các Lớp nhân ma trận, giữ nguyen FLOAT32 lúc update trọng số
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Autocast thì phải backward bằng Scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # ---- Vòng lặp THI KIỂM ĐỊNH MÙ ----
        model.eval()
        val_loss = 0.0
        corrects = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                # Biến Logits thành Số 0, 1 (Vượt dốc 0 độ = True, Âm độ = False)
                preds = (outputs > 0.0).float() 
                corrects += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = corrects / total * 100
        
        print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        
        # Lưu Trọng số nếu đạt điểm cao nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  -> ⭐ Điểm cao mới. Đã lưu bộ não 3D tốt nhất vào: {save_path}")

    print("-" * 50)
    print(f"✅ HOÀN TẤT HUẤN LUYỆN! Trọng số hoàn hảo nhất (.pth) lưu tại {save_path}")

if __name__ == "__main__":
    dataset_file = "dataset_3d_final.npz"
    if os.path.exists(dataset_file):
        train_3d_classifier(npz_file=dataset_file, epochs=40, batch_size=8)
    else:
        print(f"❌ Lỗi: Không tìm thấy file dữ liệu 3D '{dataset_file}'. Hãy chạy Data Extractor trước!")
