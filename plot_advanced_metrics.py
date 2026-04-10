import os
import sys
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from pipeline import AIPipeline

def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top: return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0

def evaluate_and_plot(images_dir, labels_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = AIPipeline(device=device)
    
    # Load Models
    path_new = 'runs_compare/runs_compare/train_yolov112/weights/best.pt'
    if not os.path.exists(path_new):
        path_new = 'runs_compare/train_yolov112/weights/best.pt'
    pipeline.load_yolo_weights(path_new)

    image_paths = glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.png"))
    
    all_gts = []
    all_preds_raw = []
    
    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        name, _ = os.path.splitext(base_name)
        label_path = os.path.join(labels_dir, name + ".txt")
        
        pil_image = Image.open(img_path).convert('RGB')
        w, h = pil_image.size
        
        gts = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy = float(parts[1])*w, float(parts[2])*h
                        bw, bh = float(parts[3])*w, float(parts[4])*h
                        gts.append([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2])
        all_gts.append(gts)
        
        # Ensemble predictions raw
        nodules_raw = []
        if pipeline.yolo_detector:
            res_main = pipeline.yolo_detector.model.predict(source=pil_image, conf=0.01, augment=True, imgsz=1024, verbose=False)
            for r in res_main:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    nodules_raw.append({'box': [x1, y1, x2, y2], 'conf': box.conf[0].item(), 'cx': (x1+x2)/2, 'cy': (y1+y2)/2})
                    
            res_clahe = pipeline.yolo_detector.model.predict(source=pipeline.apply_clahe(pil_image), conf=0.01, augment=True, imgsz=1024, verbose=False)
            for r in res_clahe:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    nodules_raw.append({'box': [x1, y1, x2, y2], 'conf': box.conf[0].item(), 'cx': (x1+x2)/2, 'cy': (y1+y2)/2})
                    
        if pipeline.yolo_aux:
            res_aux = pipeline.yolo_aux.model.predict(source=pil_image, conf=0.01, augment=True, imgsz=1024, verbose=False)
            for r in res_aux:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    nodules_raw.append({'box': [x1, y1, x2, y2], 'conf': box.conf[0].item(), 'cx': (x1+x2)/2, 'cy': (y1+y2)/2})
        
        nodules_raw = sorted(nodules_raw, key=lambda x: x['conf'], reverse=True)
        final_boxes = []
        for nodule in nodules_raw:
            is_dup = False
            for exist in final_boxes:
                dist = np.sqrt((nodule['cx'] - exist['cx'])**2 + (nodule['cy'] - exist['cy'])**2)
                if dist < 5:
                    is_dup = True; break
            if not is_dup:
                final_boxes.append(nodule)
        all_preds_raw.append(final_boxes)

    # Varying thresholds
    thresholds = np.linspace(0.01, 0.95, 50)
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in thresholds:
        tp, fp, fn = 0, 0, 0
        for i in range(len(all_gts)):
            gts = all_gts[i]
            preds = [p for p in all_preds_raw[i] if p['conf'] >= thresh]
            
            matched_gt = set()
            for p in preds:
                best_iou = 0
                best_gt = -1
                for idx, g in enumerate(gts):
                    if idx in matched_gt: continue
                    iou = calculate_iou(p['box'], g)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = idx
                if best_iou >= 0.05:
                    matched_gt.add(best_gt)
                    tp += 1
                else:
                    fp += 1
            fn += len(gts) - len(matched_gt)
            
        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (p * r) / (p + r) if p + r > 0 else 0
        
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)
        
    # --- PLOT 1: Confidence vs F1-Score & Metrics ---
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', color='#3498db', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', color='#2ecc71', linewidth=2)
    plt.plot(thresholds, f1_scores, label='F1-Score', color='#e74c3c', linewidth=2, linestyle='--')
    
    best_f1_idx = np.argmax(f1_scores)
    best_conf = thresholds[best_f1_idx]
    
    plt.axvline(x=best_conf, color='k', linestyle=':', label=f'Best Conf ({best_conf:.2f})')
    plt.title('Metrics vs Confidence Threshold (Ensemble + TTA)')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Metric Score (0 - 1.0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('confidence_curve.png', dpi=300)
    plt.close()
    
    # --- PLOT 2: Pre-defined Confusion Matrix at 0.01 ---
    # At conf 0.01 (Recall 100%)
    tp_01, fp_01, fn_01 = 0, 0, 0
    for i in range(len(all_gts)):
        gts = all_gts[i]
        preds = [p for p in all_preds_raw[i] if p['conf'] >= 0.01]
        matched_gt = set()
        for p in preds:
            best_iou = 0; best_gt = -1
            for idx, g in enumerate(gts):
                if idx in matched_gt: continue
                iou = calculate_iou(p['box'], g)
                if iou > best_iou: best_iou, best_gt = iou, idx
            if best_iou >= 0.05:
                matched_gt.add(best_gt)
                tp_01 += 1
            else:
                fp_01 += 1
        fn_01 += len(gts) - len(matched_gt)

    conf_matrix = np.array([
        [tp_01, fn_01],
        [fp_01, 0] # 0 for Background-Background which we don't count
    ])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Nodule (Pred)', 'Background (Pred)'],
                yticklabels=['Nodule (True)', 'Background (True)'],
                annot_kws={"size": 16})
    plt.title('Confusion Matrix @ Conf 0.01\n(Before Morphological 3D CNN Filter)')
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    print("Saved confidence_curve.png and confusion_matrix.png")

if __name__ == "__main__":
    evaluate_and_plot('dataset_yolo_final/images/val', 'dataset_yolo_final/labels/val')
