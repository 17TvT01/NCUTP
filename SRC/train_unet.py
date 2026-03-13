import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

# Thêm path để import model
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lung_segment import UNet

# 1. Định nghĩa DataLoader
class LungDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        
        # Chỉ lấy file có cả ảnh và mask
        self.images = []
        for f in os.listdir(img_dir):
            if f.endswith('.png'):
                mask_path = os.path.join(mask_dir, f.replace('.png', '_mask.png'))
                if os.path.exists(mask_path):
                    self.images.append(f)
                    
        self.transform_img = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor() # Tự chia / 255.0
        ])
        
        self.transform_mask = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.png', '_mask.png'))
        
        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        # Data Augmentation nhẹ chống Overfit
        if torch.rand(1) > 0.5:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)
            
        if torch.rand(1) > 0.5:
            # Random rotate nhẹ +10 to -10 độ
            angle = torch.randint(-10, 10, (1,)).item()
            img = transforms.functional.rotate(img, angle)
            mask = transforms.functional.rotate(mask, angle)

        return self.transform_img(img), self.transform_mask(mask)

# 2. Định nghĩa Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()    
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + self.smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)))
        return loss.mean()

# 3. Main Loop
def train_unet():
    # Tham số
    BATCH_SIZE = 16
    EPOCHS = 15
    LR = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Bắt đầu huấn luyện U-Net trên thiết bị: {DEVICE}")
    
    # Chuẩn bị Data
    dataset = LungDataset(
        img_dir="d:/Tool-vibecode/NCS/dataset_unet/images",
        mask_dir="d:/Tool-vibecode/NCS/dataset_unet/masks",
        img_size=256
    )
    
    # Chia Train (90%) - Val (10%)
    train_sz = int(0.9 * len(dataset))
    val_sz = len(dataset) - train_sz
    train_data, val_data = torch.utils.data.random_split(dataset, [train_sz, val_sz])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Tổng số mẫu: {len(dataset)} | Train: {train_sz} | Val: {val_sz}")
    
    # Khởi tạo Model
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    
    # Hàm Loss = Cân bằng giữa Điểm ảnh (BCE) và Hình dạng cục bộ (DiceLoss)
    criterion_bce = nn.BCEWithLogitsLoss() 
    criterion_dice = DiceLoss()
    def combined_loss(pred_logits, true_mask):
        pred_sigmoid = torch.sigmoid(pred_logits)
        bce = criterion_bce(pred_logits, true_mask)
        dice = criterion_dice(pred_sigmoid, true_mask)
        return bce + dice
        
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    
    # Bắt đầu Train
    best_loss = float('inf')
    os.makedirs('d:/Tool-vibecode/NCS/weights', exist_ok=True)
    
    for epoch in range(1, EPOCHS + 1):
        # ---------------- TRAINING ----------------
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [TRAIN]")
        for imgs, masks in loop:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Khác với code cũ U-Net tự Sigmoid, Loss Combined yêu cầu Logits đầu vào
            # Sẽ cần sửa code U-Net bỏ sigmoid lớp cuối đi khi Train
            if scaler:
                with torch.amp.autocast('cuda'):
                    pred = model(imgs) # Lưu ý: Model trả về sigmoid rồi
                    # Nhưng Combined loss nhận Logits, ta hack nhẹ:
                    # Hoặc là sửa UNet, hoặc truyền tay:
                    # Vì tớ không muốn sửa code UNet, ta dùng MSE hoặc Dice thẳng.
                    pass
            pass

def get_data_loaders(batch_size=8):
    dataset = LungDataset(img_dir="dataset_unet/images", mask_dir="dataset_unet/masks")
    print(f"Tổng số mẫu tìm thấy: {len(dataset)}")
    
    if len(dataset) == 0:
        raise ValueError("Thư mục dataset_unet trống không! Hãy kiểm tra script tạo dữ liệu.")
        
    train_sz = int(0.9 * len(dataset))
    val_sz = len(dataset) - train_sz
    train_data, val_data = torch.utils.data.random_split(dataset, [train_sz, val_sz])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def train_unet_v2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Bắt đầu huấn luyện U-Net trên: {device}")
    
    train_loader, val_loader = get_data_loaders(batch_size=8)
    
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # Tắt hoàn toàn Scaler vì AMP hay gây lỗi NaN trên kiến trúc nhỏ
    scaler = None
    
    dice = DiceLoss()
    mse = nn.MSELoss() # Dùng MSELoss thay thế BCE do Autocast không thích BCELoss với Sigmoid
    
    best_loss = float('inf')
    os.makedirs("weights", exist_ok=True)
    
    for epoch in range(1, 16):
        model.train()
        t_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/15")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            
            # Train Float32 truyền thống siêu an toàn
            preds = model(imgs)
            loss_mse = mse(preds, masks)
            loss_dice = dice(preds, masks)
            loss = loss_mse + loss_dice
            loss.backward()
            
            # Clip gradient siêu an toàn chống nổ
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
                
            t_loss += loss.item()
            pbar.set_postfix(Loss=loss.item())
            
        avg_loss = t_loss / len(train_loader)
        print(f"Average Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Đảm bảo lưu đúng đường dẫn
            torch.save(model.state_dict(), "weights/unet_best.pth")
            print("=> Đã lưu mô hình Weights/unet_best.pth tốt nhất!")

if __name__ == '__main__':
    train_unet_v2()
