from ultralytics import YOLO
import os

class NoduleDetector:
    """
    Sử dụng YOLO11n (Bản nano siêu phân giải & nhạy của YOLO11) để thực hiện Object Detection.
    Mô hình này sẽ nhận ảnh Đầu vào (ảnh gốc hoặc ảnh đã qua U-Net) và xuất ra 
    các Bounding Box (hộp giới hạn) nghi ngờ là Nốt phổi (Nodule).
    """
    def __init__(self, weights_path="weights/yolo11n.pt", device='cpu'):
        self.device = device
        
        # Nếu chưa có weights. YOLO sẽ tự động tải file yolo11n.pt về từ Internet
        self.model = YOLO(weights_path)
        print("Đã tải xong module YOLO11n Nodule Detector!")

    def predict(self, pil_image, conf_threshold=0.25):
        """
        Dự đoán vị trí nốt phổi trên ảnh PIL
        :param pil_image: Ảnh PIL đầu vào
        :param conf_threshold: Ngưỡng tự tin tối thiểu (0.25 = 25%)
        :return: Danh sách các nốt phát hiện được định dạng chuẩn
        """
        # YOLO11 hỗ trợ infer trực tiếp trên ảnh PIL Image
        # verbose=False để tránh in quá nhiều log làm ô nhiễm console
        results = self.model.predict(source=pil_image, conf=conf_threshold, device=self.device, verbose=False)
        
        detected_nodules = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Lấy toạ độ chuẩn
                x1, y1, x2, y2 = box.xyxy[0].tolist() 
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                
                # Tính toạ độ tâm X, Y của nốt
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Tính xấp xỉ thể tích / diện tích nốt (Voxel) bằng độ rộng x chiều cao box
                width = int(x2 - x1)
                height = int(y2 - y1)
                approx_voxel = width * height
                
                detected_nodules.append({
                    "x1": int(x1), "y1": int(y1), 
                    "x2": int(x2), "y2": int(y2),
                    "center_x": center_x,
                    "center_y": center_y,
                    "voxel": approx_voxel,
                    "confidence": conf,
                    "class_id": cls_id
                })
                
        return detected_nodules
