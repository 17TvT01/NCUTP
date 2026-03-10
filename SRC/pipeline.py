import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from models.lung_segment import load_unet_model
from models.nodule_detect import NoduleDetector
from models.nodule_segment import fallback_segmentation
import cv2

class AIPipeline:
    def __init__(self, device='cpu'):
        self.device = device
        self.unet_model = load_unet_model(device=device)
        print("Đã tải module UNet Segmentation!")
        
        # Hàm __init__ khởi tạo tự động file YOLO weights
        self.yolo_detector = NoduleDetector(weights_path="yolov8n.pt", device=device)
        print("Đã tải module YOLOv8n Detection!")
        
        # Các phép biến đổi tiêu chuẩn khi đưa ảnh vào model
        self.transform = transforms.Compose([
            transforms.ToTensor(), # Chuyển image thành mảng Float tensor [0.0, 1.0]
        ])

    def load_yolo_weights(self, weights_path):
        """Khởi tạo lại Detector khi người dùng đổi file weights mới (.pt)"""
        self.yolo_detector = NoduleDetector(weights_path=weights_path, device=self.device)
        print(f"Đã tải mô hình YOLOv8 mới từ: {weights_path}")

    def preprocess_image(self, pil_image):
        # Resize về đúng input của U-Net (256x256 để mô hình chạy cực nhẹ)
        img_resized = pil_image.resize((256, 256), Image.Resampling.BILINEAR)
        img_tensor = self.transform(img_resized)
        
        # Thêm batch dimension (1, C, H, W) -> Vì ảnh DICOM gray scale nên C=1.
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        return img_tensor, pil_image.size

    def predict_lung_mask(self, img_tensor):
        """ Dự đoán vùng phổi, trả ra mask nhị phân """
        with torch.no_grad(): # Tắt tính gradient để chạy nhanh và đỡ tốn RAM
            output = self.unet_model(img_tensor)
            mask = output > 0.5 # Ngưỡng xác suất 0.5 để quyết định đâu là phổi
            mask = mask.float().cpu().numpy()[0, 0, :, :] # Lấy array 2D
        return mask

    def run_full_pipeline(self, pil_image):
        # 1. Tiền xử lý
        img_tensor, original_size = self.preprocess_image(pil_image)
        
        # 2. Bước 1: Lung Segmentation
        # Lấy mask nhị phân từ U-Net
        lung_mask = self.predict_lung_mask(img_tensor)
        
        # Áp dụng Mask phổi (tạm thời lấp đầy để demo U-Net, do U-net siêu nhẹ hiện tại chưa đủ chuẩn, ta sẽ dùng numpy resize mask)
        mask_resized = cv2.resize(lung_mask.astype(np.uint8), original_size)
        
        # 3. Bước 2: YOLO Detection (Cắt Nodule)
        nodules_raw = self.yolo_detector.predict(pil_image, conf_threshold=0.25)
        
        # BỘ LỌC CHẶN BUÔNG (Phòng trường hợp YOLO phớt lờ tham số conf)
        safe_nodules = [n for n in nodules_raw if n['confidence'] >= 0.25]
        
        # Sắp xếp các nốt tìm được theo độ tin cậy (Confidence) giảm dần
        safe_nodules = sorted(safe_nodules, key=lambda x: x['confidence'], reverse=True)
        
        # LỌC NHIỄU YOLO VÀ CUSTOM NMS (GỘP BOX):
        nodules = []
        max_nodule_size = original_size[0] * 0.25 # Nốt không vượt quá 25% chiều rộng phổi
        max_nodules_per_slice = 5 # Không lấy quá 5 nốt/1 lát cắt để chống nhiễu tuyệt đối
        
        for nodule in safe_nodules:
            if len(nodules) >= max_nodules_per_slice:
                break
                
            x1, y1, x2, y2 = nodule["x1"], nodule["y1"], nodule["x2"], nodule["y2"]
            width = x2 - x1
            height = y2 - y1
            
            # Tính tỷ lệ khung hình nốt phổi (Nốt thật thường có cục xấp xỉ hình cầu)
            aspect_ratio = width / height if height > 0 else 0
            
            if width < max_nodule_size and height < max_nodule_size:
                # Siết chặt lại! Nốt thật không bao giờ dẹp lép (1 chiều dài hơn 2 lần chiều kia)
                if 0.5 <= aspect_ratio <= 2.0:
                    # Kiểm tra trùng lặp (Custom Non-Maximum Suppression)
                    is_duplicate = False
                    for existing in nodules:
                        dist = np.sqrt((nodule["center_x"] - existing["center_x"])**2 + (nodule["center_y"] - existing["center_y"])**2)
                        # Nếu Box mới sinh ra nằm cách Box cũ dưới 15 pixel thì chắc chắn là vẽ đè lên cùng 1 nốt
                        if dist < 15: 
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        nodules.append(nodule)
        
        # 4. Bước 3: Khoanh vùng chính xác (Nodule Segmentation)
        # Thay vì chỉ vẽ Box chữ nhật, ta cắt nốt ra và tìm hình dạng chính xác cực kỳ chi tiết
        cv_img_gray = np.array(pil_image)
        for nodule in nodules:
            # Lấy vùng crop của Bounding Box
            x1, y1, x2, y2 = nodule["x1"], nodule["y1"], nodule["x2"], nodule["y2"]
            patch = cv_img_gray[y1:y2, x1:x2]
            
            # Nếu patch hợp lệ
            if patch.size > 0:
                # Tìm mask hình chi tiết (true/false)
                fine_mask = fallback_segmentation(patch)
                # Lưu dưới dạng np.array để chuyển về giao diện vẽ
                nodule["fine_mask"] = fine_mask
            else:
                nodule["fine_mask"] = None
        
        result = {
            "mask": lung_mask,
            "nodules": nodules, # Danh sách chi tiết tọa độ và độ tự tin
            "message": f"Tìm thấy {len(nodules)} nốt phổi trên lát cắt."
        }
        return result
