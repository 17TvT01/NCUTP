import torch
import torch.nn as nn
from models.lung_segment import UNet
import numpy as np
import cv2
import math

class MiniUNet(nn.Module):
    """
    Kế thừa kiến trúc U-Net 2D siêu nhẹ từ Lung Segmentation nhưng tập trung
    huấn luyện dành riêng cho các patch (vùng cắt nhỏ) của nốt phổi.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(MiniUNet, self).__init__()
        # Để tiết kiệm code và RAM, chúng ta có thể tái sử dụng luôn class UNet đã viết
        # Hoặc viết một mạng giản lược hơn nữa nếu muốn. 
        # Tạm thời tái sử dụng kiến trúc UNet nhẹ bằng cách gọi instance của nó 
        self.net = UNet(in_channels, out_channels)
        
    def forward(self, x):
        return self.net(x)

def load_nodule_segment_model(weights_path=None, device='cpu'):
    model = MiniUNet()
    if weights_path and torch.cuda.is_available():
        pass # TODO: Viết logic nạp weights cụ thể
    model.to(device)
    model.eval()
    return model

def fallback_segmentation(patch_img_2d):
    """
    Sử dụng thuật toán xử lý ảnh cổ điển (Otsu Thresholding) + ConvexHull 
    để tạo liền mạch nốt phổi, tránh bị tách rời làm 2 đốm.
    Tính toán thêm Diện Tích và Độ Tròn để chống nhiễu Mạch Máu/Xương.
    """
    # 0. Tăng cường độ tương phản (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(patch_img_2d)

    # 1. Thresholding: Nốt phổi sáng hơn nền phổi đen, OTSU sẽ phân cụm hoàn hảo.
    blur = cv2.GaussianBlur(enhanced, (3,3), 0)
    _, binary_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. Xóa cát nhiễu (Open) và lấp đầy các khoảng trống bên trong nốt (Close)
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Tách Contour theo ngoại vi
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(patch_img_2d)
    
    h, w = patch_img_2d.shape
    center_patch = (w // 2, h // 2)
    max_dist_allowed = max(h, w) * 0.40 # Lấy các mảnh vụn cách tâm Box không quá xa
    
    valid_points = []
    
    # Biến lưu trữ thông số Hình thái
    morph_area = 0.0
    morph_circularity = 0.0
    
    if contours:
        for c in contours:
            # Bỏ mạt bụi quá nhỏ
            if cv2.contourArea(c) < 3:
                continue
                
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                dist = np.sqrt((cX - center_patch[0])**2 + (cY - center_patch[1])**2)
                
                # Nếu mảnh này thuộc về nốt tâm trung tâm
                if dist < max_dist_allowed:
                    valid_points.extend(c) # Gom tất cả đỉnh vào mảng duy nhất
        
        # 3. Thuật toán bao lưới ConvexHull giúp chắp ghép các chấm rời của một khối u thành cục rắn chắn
        if valid_points:
            valid_points = np.array(valid_points)
            hull = cv2.convexHull(valid_points)
            cv2.drawContours(final_mask, [hull], -1, 255, thickness=cv2.FILLED)
            
            # ĐO ĐẠC HÌNH THÁI HỌC CỦA KHỐI BAO NÀY
            morph_area = cv2.contourArea(hull)
            perimeter = cv2.arcLength(hull, True)
            
            if perimeter > 0:
                 # Công thức tính độ tròn (1.0 là tròn tuyệt đối)
                 morph_circularity = 4 * math.pi * morph_area / (perimeter * perimeter)
             
    return final_mask > 0, morph_area, morph_circularity
