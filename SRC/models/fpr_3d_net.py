import torch
import torch.nn as nn

class Lightweight3DCNN(nn.Module):
    """
    Mạng Xử lý 3 Chiều siêu nhẹ (FPR Classifier).
    Đầu vào: Khối Tensor chắp ghép 3D (Batch_Size, Channel=1, Depth=16, Height=32, Width=32)
    Đầu ra: Xác suất Nốt Phổi Thật (Sigmoid 0 -> 1)
    """
    def __init__(self):
        super(Lightweight3DCNN, self).__init__()
        
        # Block 1: Trích xuất viền cơ bản (Cạnh, khối mờ)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            # Giảm kích thước còn 8x16x16
            nn.MaxPool3d(kernel_size=2, stride=2) 
        )
        
        # Block 2: Tìm kiếm cấu trúc Khối Cầu (Mass)
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # Giảm kích thước còn 4x8x8
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Block 3: Phân rã cấu trúc Mạch máu (Đường ống méo)
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # Global Average Pooling: Biến khối 3D (4x8x8) thành 1 điểm dữ liệu (1x1x1) => Cứu 90% bộ nhớ RAM
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Bộ phân loại quyết định (Classifier Head)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3), # Tránh học vẹt nốt xương
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x.shape = (B, 1, 16, 32, 32)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        # Duỗi siêu tensor thành vector phẳng (Dẹt lại để ném vào bộ Linear)
        out = out.view(out.size(0), -1) 
        
        # Đi qua bộ Fully Connected
        logits = self.classifier(out)
        return logits # Dòng ném Sigmoid sẽ thực thi bên module Loss/Inference

if __name__ == "__main__":
    # Test thử sức chịu đựng của Mạng (Inference Dummy 1 khối)
    model = Lightweight3DCNN()
    dummy_input = torch.randn(1, 1, 16, 32, 32) # (Batch, C, D, H, W)
    output = model(dummy_input)
    
    print(f"Kiến trúc Mạng FPR 3D đã được tạo thành công!")
    print(f"Tổng số tham số: {sum(p.numel() for p in model.parameters())}")
    print(f"Output Shape: {output.shape} (Chờ Sigmoid để ép ra 0% -> 100%)")
