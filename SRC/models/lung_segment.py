import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    Kiến trúc U-Net 2D siêu nhẹ.
    Được thiết kế tối giản hoá số lượng feature maps để có thể chạy mượt trên CPU máy yếu.
    Dùng để tách (segment) vùng phổi từ ảnh chụp CT gốc.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Encoder (Downsampling)
        self.enc1 = conv_block(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = conv_block(32, 64)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(64, 128)

        # Decoder (Upsampling)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = conv_block(128, 64) # 128 vì concat skip connection 64 + upconv 64
        
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = conv_block(64, 32)
        
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = conv_block(32, 16)

        # Output layer
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Đường mã hóa
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Đáy chữ U
        b = self.bottleneck(self.pool3(e3))
        
        # Đường giải mã
        d3 = self.upconv3(b)
        d3 = torch.cat((e3, d3), dim=1) # Ghép nối (Skip Connection)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        
        out = self.out_conv(d1)
        return self.sigmoid(out)

import os

# Hàm khởi tạo hoặc nạp trọng số
def load_unet_model(weights_path=None, device='cpu'):
    model = UNet()
    
    # Load Weights nếu có truyền đường dẫn và file tồn tại
    if weights_path and os.path.exists(weights_path):
        import traceback
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
            print(f"🟢 Đã ráp não cho U-Net thành công từ: {weights_path}")
        except Exception as e:
            print(f"🔴 Lỗi khi ráp não U-Net: {e}")
            traceback.print_exc()
    else:
        print(f"⚠️ U-Net đang chạy bằng bản năng gốc (Random Weights)! Không tìm thấy: {weights_path}")
        
    model.to(device)
    model.eval() # Chế độ suy luận
    return model
