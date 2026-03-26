from ultralytics import YOLO
import json
import os

print('🚀 Bắt đầu quá trình đánh giá (Evaluation) 2 mô hình...', flush=True)

metrics_report = {}

# 1. Evaluate YOLOv8 (best_yolo_medical.pt)
try:
    print('\n' + '='*50)
    print('--- Đánh giá YOLOv8 (best_yolo_medical.pt) ---', flush=True)
    model_v8 = YOLO('best_yolo_medical.pt')
    res_v8 = model_v8.val(
        data='dataset_yolo_final/data.yaml',
        imgsz=640,
        plots=True,
        project='runs_compare/evaluate',
        name='yolov8_val',
        exist_ok=True
    )
    
    metrics_report['yolov8'] = {
        'mAP50': res_v8.box.map50,
        'mAP50-95': res_v8.box.map,
        'precision': res_v8.box.mp,  # mean precision
        'recall': res_v8.box.mr,     # mean recall
        'fitness': res_v8.box.fitness()
    }
    print(f"[YOLOv8] mAP50={res_v8.box.map50:.4f}, mAP50-95={res_v8.box.map:.4f}, P={res_v8.box.mp:.4f}, R={res_v8.box.mr:.4f}", flush=True)

except Exception as e:
    print('❌ Lỗi đánh giá YOLOv8:', e)

# 2. Evaluate YOLOv11 (runs_compare/train_yolov11/weights/best.pt)
try:
    print('\n' + '='*50)
    print('--- Đánh giá YOLOv11 (train_yolov11) ---', flush=True)
    model_v11_path = 'runs_compare/train_yolov11/weights/best.pt'
    if not os.path.exists(model_v11_path):
        print(f"Không tìm thấy file {model_v11_path}. Vui lòng kiểm tra lại quá trình huấn luyện.")
    else:
        model_v11 = YOLO(model_v11_path)
        res_v11 = model_v11.val(
            data='dataset_yolo_final/data.yaml',
            imgsz=640,
            plots=True,
            project='runs_compare/evaluate',
            name='yolov11_val',
            exist_ok=True
        )
        
        metrics_report['yolov11'] = {
            'mAP50': res_v11.box.map50,
            'mAP50-95': res_v11.box.map,
            'precision': res_v11.box.mp,
            'recall': res_v11.box.mr,
            'fitness': res_v11.box.fitness()
        }
        print(f"[YOLOv11] mAP50={res_v11.box.map50:.4f}, mAP50-95={res_v11.box.map:.4f}, P={res_v11.box.mp:.4f}, R={res_v11.box.mr:.4f}", flush=True)

except Exception as e:
    print('❌ Lỗi đánh giá YOLOv11:', e)

# Export kết quả ra JSON
with open('report/metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics_report, f, indent=4)

print('\n✅ Đánh giá hoàn tất! Kết quả đã được lưu tại report/metrics.json', flush=True)
