import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import glob
from PIL import Image
import cv2
import numpy as np
import torch
from models.nodule_detect import NoduleDetector

def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def apply_clahe(pil_img):
    img_np = np.array(pil_img)
    if len(img_np.shape) == 3:
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_np
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(img_gray)
    if len(img_np.shape) == 3:
         cl = cv2.cvtColor(cl, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(cl)

def analyze_ensemble(images_dir, labels_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Init 3 models
    path1 = "runs_compare/runs_compare/train_yolov112/weights/best.pt"
    if not os.path.exists(path1):
        path1 = "runs_compare/train_yolov112/weights/best.pt"
    model1 = NoduleDetector(path1, device=device) # Trained on Crop
    model2 = NoduleDetector("best.pt", device=device) # Trained on full
    model3 = NoduleDetector("runs_compare/train_yolov11/weights/best.pt", device=device) # Trained on full?
    
    image_paths = glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.png"))
    
    total_gt = 0
    matched_tta = 0
    
    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        name, _ = os.path.splitext(base_name)
        label_path = os.path.join(labels_dir, name + ".txt")
        
        pil_image = Image.open(img_path).convert('RGB')
        w, h = pil_image.size
        
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx = float(parts[1]) * w
                        cy = float(parts[2]) * h
                        bw = float(parts[3]) * w
                        bh = float(parts[4]) * h
                        gt_boxes.append([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2])
        
        total_gt += len(gt_boxes)
        if len(gt_boxes) == 0: continue
        
        raw_boxes = []
        
        # 1. Model 1 (Cropped model) BUT we just pass full image with TTA
        res1 = model1.model.predict(source=pil_image, conf=0.01, augment=True, imgsz=1024, verbose=False)
        for r in res1:
            for box in r.boxes:
                raw_boxes.append(box.xyxy[0].tolist())
                
        # 2. Model 1 with CLAHE
        res1_c = model1.model.predict(source=apply_clahe(pil_image), conf=0.01, augment=True, imgsz=1024, verbose=False)
        for r in res1_c:
            for box in r.boxes:
                raw_boxes.append(box.xyxy[0].tolist())
                
        # 3. Model 2 (Old best)
        res2 = model2.model.predict(source=pil_image, conf=0.01, augment=True, imgsz=1024, verbose=False)
        for r in res2:
            for box in r.boxes:
                raw_boxes.append(box.xyxy[0].tolist())
                
        # 4. Model 3 (Medical)
        try:
            res3 = model3.model.predict(source=pil_image, conf=0.01, augment=True, imgsz=640, verbose=False)
            for r in res3:
                for box in r.boxes:
                    raw_boxes.append(box.xyxy[0].tolist())
        except:
            pass

        # Evaluate combined raw recall
        matched_gt = set()
        for p_box in raw_boxes:
            best_iou = 0
            best_gt_idx = -1
            for idx, g_box in enumerate(gt_boxes):
                if idx in matched_gt: continue
                iou = calculate_iou(p_box, g_box)
                if iou > 0.05:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou >= 0.05:
                matched_gt.add(best_gt_idx)
                
        matched_tta += len(matched_gt)
        
    print(f"Total GT Boxes: {total_gt}")
    print(f"ENSEMBLE RAW Matched: {matched_tta} / {total_gt} ({matched_tta/total_gt*100:.2f}%)")

if __name__ == '__main__':
    analyze_ensemble('dataset_yolo_final/images/val', 'dataset_yolo_final/labels/val')
