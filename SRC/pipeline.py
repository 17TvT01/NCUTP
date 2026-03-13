import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from models.lung_segment import load_unet_model
from models.nodule_detect import NoduleDetector
from models.nodule_segment import fallback_segmentation
from models.fpr_3d_net import Lightweight3DCNN
import cv2

class AIPipeline:
    def __init__(self, device='cpu'):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.unet_model = load_unet_model(weights_path="weights/unet_best.pth", device=self.device)
        print("Đã tải module UNet Segmentation!")
        
        # Hàm __init__ khởi tạo tự động file YOLO weights
        self.yolo_detector = NoduleDetector(weights_path="yolov8n.pt", device=device)
        print("Đã tải module YOLOv8n Detection!")
        
        # Hàm nạp Não Phân Loại 3 Chiều FPR
        self.fpr_model = Lightweight3DCNN().to(device)
        fpr_weights = "weights/fpr_3d_best.pth"
        if os.path.exists(fpr_weights):
            self.fpr_model.load_state_dict(torch.load(fpr_weights, map_location=device))
            self.fpr_model.eval()
            print("🟢 Đã kích hoạt radar 3D phân loại Mạch Máu/Xương!")
        else:
            print("⚠️ Cảnh báo: Chưa tìm thấy weights/fpr_3d_best.pth. Cần train AI 3D trước!")
            self.fpr_model = None

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

    def run_full_pipeline(self, pil_image, full_volume=None, slice_idx=0,
                          conf_threshold=0.25, min_voxel=50, fpr_threshold=0.50):
        # 1. Tiền xử lý
        img_tensor, original_size = self.preprocess_image(pil_image)
        
        # 2. Bước 1: Lung Segmentation
        # Lấy mask nhị phân từ U-Net
        lung_mask = self.predict_lung_mask(img_tensor)
        
        # Áp dụng Mask phổi (tạm thời lấp đầy để demo U-Net, do U-net siêu nhẹ hiện tại chưa đủ chuẩn, ta sẽ dùng numpy resize mask)
        mask_resized = cv2.resize(lung_mask.astype(np.uint8), original_size)
        
        # 3. Bước 2: YOLO Detection (Cắt Nodule)
        nodules_raw = self.yolo_detector.predict(pil_image, conf_threshold=conf_threshold)
        
        # BỘ LỌC CHẶN BUÔNG (Phòng trường hợp YOLO phớt lờ tham số conf)
        safe_nodules = [n for n in nodules_raw if n['confidence'] >= conf_threshold]
        
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
        
        # 4. Bước 3: Khoanh vùng chính xác (Nodule Segmentation) VÀ HẬU XỬ LÝ LỌC NHIỄU (Morphological Filter)
        cv_img_gray = np.array(pil_image)
        
        final_nodules = [] # Danh sách nốt thực sự được Phê duyệt cuối cùng
        
        for nodule in nodules:
            # Lấy vùng crop của Bounding Box
            x1, y1, x2, y2 = nodule["x1"], nodule["y1"], nodule["x2"], nodule["y2"]
            patch = cv_img_gray[y1:y2, x1:x2]
            
            # Nếu patch hợp lệ
            if patch.size > 0:
                # Tìm mask hình chi tiết (true/false) VÀ Đo lường đặc tính Hình học Kích Thước/Độ Tròn
                fine_mask, morph_area, morph_circ = fallback_segmentation(patch)
                
                # BỘ LỌC CỨNG (Hậu Xử Lý Hình Thái Học): Trảm Xương & Mạch Máu Cực Đoan
                # Vì các Nốt lớn (Mass) có khi lên tới diện tích 2000-3000 pixel, ta phải nới lỏng Area > 2000.
                # Độ tròn Circularity cũng phải hạ xuống 0.15 để tránh "giết nhầm" nốt ung thư dị hình.
                if morph_circ < 0.15 or morph_area > 2000 or morph_area < min_voxel:
                    continue # Vứt bỏ vì là sợi mạch máu dài quá mức hoặc khúc xương sườn to qúa mức
                    
                # BỘ LỌC TỐI CAO - 3D CNN CLASSIFIER (False Positive Reduction)
                # Đưa nốt này vào ngắm dưới lăng kính không gian 3 chiều
                fpr_score = 1.0 # Mặc định pass
                if self.fpr_model is not None and full_volume is not None:
                    # 1. Xác định tọa độ Mỏ neo (Tâm)
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                    
                    # 2. Xây Hộp hộp Rubik 3D (Z: 16, H: 32, W: 32)
                    D, H_vol, W_vol = len(full_volume), full_volume[0].height, full_volume[0].width
                    pD, pH, pW = 16, 32, 32
                    
                    z_start = max(0, slice_idx - pD//2)
                    z_end = min(D, z_start + pD)
                    y_start = max(0, cy - pH//2)
                    y_end = min(H_vol, y_start + pH)
                    x_start = max(0, cx - pW//2)
                    x_end = min(W_vol, x_start + pW)
                    
                    volume_slice = []
                    for s in range(z_start, z_end):
                       gray = np.array(full_volume[s])
                       patch_2d = gray[y_start:y_end, x_start:x_end]
                       volume_slice.append(patch_2d)
                    patch_3d_np = np.stack(volume_slice) # (D, H, W)
                    
                    if patch_3d_np.shape != (pD, pH, pW):
                        pad_z = (0, max(0, pD - patch_3d_np.shape[0]))
                        pad_y = (0, max(0, pH - patch_3d_np.shape[1]))
                        pad_x = (0, max(0, pW - patch_3d_np.shape[2]))
                        patch_3d_np = np.pad(patch_3d_np, (pad_z, pad_y, pad_x), mode='constant')
                        
                    # 3. Phán xét bằng Não bộ Voxel
                    # Scale theo PyTorch (Batch 1, Channel 1) -> / 255.0
                    tensor_3d = torch.tensor(patch_3d_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
                    tensor_3d = tensor_3d.to(self.device)
                    
                    with torch.no_grad():
                         with torch.amp.autocast('cuda' if self.device.type=='cuda' else 'cpu'):
                             logits = self.fpr_model(tensor_3d)
                             prob = torch.sigmoid(logits).item()
                             fpr_score = prob
                             
                    # CHỐT HẠ: Nếu mô hình 3D bảo Tỷ lệ nốt phổi thấp hơn 50% => Gạch bỏ (Mạch Máu/Xương giả mạo!)
                    if fpr_score < fpr_threshold:
                        continue
                        
                # Lưu Dữ Liệu vào nốt hợp lệ (Cùng với tỷ lệ ung thư 3D)
                nodule["fine_mask"] = fine_mask
                nodule["morph_area"] = morph_area
                nodule["morph_circ"] = morph_circ
                nodule["fpr_score"] = fpr_score
                final_nodules.append(nodule)
            else:
                pass # Patch lỗi, vứt
        
        result = {
            "mask": lung_mask,
            "nodules": final_nodules, # Danh sách chi tiết nốt ĐÃ ĐƯỢC CHẮT LỌC KỸ LƯỠNG NHẤT
            "message": f"Tìm thấy {len(final_nodules)} nốt phổi trên lát cắt."
        }
        return result
