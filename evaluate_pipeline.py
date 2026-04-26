import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import glob
from PIL import Image
import numpy as np
import torch
from pipeline import AIPipeline
import json

def calculate_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
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

def evaluate_pipeline_on_dataset(pipeline, images_dir, labels_dir, conf_threshold=0.05, iou_thresh=0.1):
    image_paths = glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.png"))
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
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
                        cx_norm = float(parts[1])
                        cy_norm = float(parts[2])
                        w_norm = float(parts[3])
                        h_norm = float(parts[4])
                        
                        cx, cy = cx_norm * w, cy_norm * h
                        bw, bh = w_norm * w, h_norm * h
                        gt_boxes.append([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2])
                        
        # PIPELINE ENSEMBLE TTA (Bypass U-Net bounding offset since image is already cropped)
        nodules_raw = []
        conf_target = 0.01 
        
        if pipeline.yolo_detector:
            res_main = pipeline.yolo_detector.model.predict(source=pil_image, conf=conf_target, augment=True, imgsz=1024, verbose=False)
            for r in res_main:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    nodules_raw.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'center_x': (x1+x2)/2, 'center_y': (y1+y2)/2, 'confidence': box.conf[0].item()})
            
            res_clahe = pipeline.yolo_detector.model.predict(source=pipeline.apply_clahe(pil_image), conf=conf_target, augment=True, imgsz=1024, verbose=False)
            for r in res_clahe:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    nodules_raw.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'center_x': (x1+x2)/2, 'center_y': (y1+y2)/2, 'confidence': box.conf[0].item()})
                    
        if pipeline.yolo_aux:
            res_aux = pipeline.yolo_aux.model.predict(source=pil_image, conf=conf_target, augment=True, imgsz=1024, verbose=False)
            for r in res_aux:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    nodules_raw.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'center_x': (x1+x2)/2, 'center_y': (y1+y2)/2, 'confidence': box.conf[0].item()})
                    
        # NMS Filter logic
        safe_nodules = [n for n in nodules_raw if n['confidence'] >= conf_threshold]
        safe_nodules = sorted(safe_nodules, key=lambda x: x['confidence'], reverse=True)
        
        final_boxes = []
        for nodule in safe_nodules:
            final_boxes.append([nodule['x1'], nodule['y1'], nodule['x2'], nodule['y2']])
                
        # Matching
        matched_gt = set()
        for p_box in final_boxes:
            best_iou = 0
            best_gt_idx = -1
            for idx, g_box in enumerate(gt_boxes):
                if idx in matched_gt: continue
                iou = calculate_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou >= iou_thresh:
                matched_gt.add(best_gt_idx)
                tp_total += 1
            else:
                fp_total += 1
                
        # False negatives
        fn_total += (len(gt_boxes) - len(matched_gt))
        
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    
    return {
        'TP': tp_total,
        'FP': fp_total,
        'FN': fn_total,
        'Precision': precision,
        'Recall': recall
    }

if __name__ == '__main__':
    print('🚀 Khởi tạo AI Pipeline...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = AIPipeline(device=device)
    
    images_dir = 'dataset_yolo_final/images/val'
    labels_dir = 'dataset_yolo_final/labels/val'
    
    print('\n=======================================')
    print('1. Đánh giá App với mô hình CŨ (best.pt)')
    print('=======================================')
    pipeline.load_yolo_weights('best.pt')
    metrics_old = evaluate_pipeline_on_dataset(pipeline, images_dir, labels_dir, conf_threshold=0.01, iou_thresh=0.05)
    print(f"Metrics (best.pt): {json.dumps(metrics_old, indent=2)}")
    
    print('\n=======================================')
    print('2. Đánh giá App với mô hình MỚI (runs_compare/.../best.pt)')
    print('=======================================')
    path_new = 'runs_compare/runs_compare/train_yolov112/weights/best.pt'
    if not os.path.exists(path_new):
        path_new = 'runs_compare/train_yolov112/weights/best.pt' # fallback
    if not os.path.exists(path_new):
        path_new = 'runs_compare/train_yolov11/weights/best.pt'
    pipeline.load_yolo_weights(path_new)
    metrics_new = evaluate_pipeline_on_dataset(pipeline, images_dir, labels_dir, conf_threshold=0.01, iou_thresh=0.05)
    print(f"Metrics (New Model): {json.dumps(metrics_new, indent=2)}")
    
    print('\n✅ Hoàn tất đánh giá PIPELINE!')
