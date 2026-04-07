from ultralytics import YOLO
import json
import os

print('🚀 Bắt đầu quá trình đánh giá lại với CẤU HÌNH CONFIDENCE THẤP (conf=0.05)...', flush=True)

metrics_report = {}

# 1. Evaluate YOLOv11 (Old/GPU 1 - best.pt ở rễ)
try:
    print('\n' + '='*50)
    print('--- Đánh giá YOLOv11 (best.pt ở thư mục gốc) ở conf=0.05 ---', flush=True)
    model_old = YOLO('best.pt')
    res_old = model_old.val(
        data='dataset_yolo_final/data.yaml',
        imgsz=640,
        plots=False,
        project='runs_compare/evaluate',
        name='yolov11_gpu1_val_conf05',
        exist_ok=True,
        conf=0.05  # Giảm mức độ tự tin để bắt được nhiều nốt nhất có thể
    )
    
    metrics_report['yolov11_gpu1_conf05'] = {
        'mAP50': res_old.box.map50,
        'mAP50-95': res_old.box.map,
        'precision': res_old.box.mp,
        'recall': res_old.box.mr,
    }
    print(f"[YOLOv11 - best.pt] mAP50={res_old.box.map50:.4f}, mAP50-95={res_old.box.map:.4f}, P={res_old.box.mp:.4f}, R={res_old.box.mr:.4f}", flush=True)

except Exception as e:
    print('❌ Lỗi đánh giá YOLOv11 (best.pt):', e)

# 2. Evaluate YOLOv11 (New/GPU 2 - runs_compare/train_yolov11/weights/best.pt)
try:
    print('\n' + '='*50)
    print('--- Đánh giá YOLOv11 (runs_compare/train_yolov11/weights/best.pt) ở conf=0.05 ---', flush=True)
    model_new_path = 'runs_compare/train_yolov11/weights/best.pt'
    if not os.path.exists(model_new_path):
        print(f"Không tìm thấy file {model_new_path}.")
    else:
        model_new = YOLO(model_new_path)
        res_new = model_new.val(
            data='dataset_yolo_final/data.yaml',
            imgsz=640,
            plots=False,
            project='runs_compare/evaluate',
            name='yolov11_gpu2_val_conf05',
            exist_ok=True,
            conf=0.05  # Giảm mức độ tự tin
        )
        
        metrics_report['yolov11_gpu2_conf05'] = {
            'mAP50': res_new.box.map50,
            'mAP50-95': res_new.box.map,
            'precision': res_new.box.mp,
            'recall': res_new.box.mr,
        }
        print(f"[YOLOv11 - runs_compare] mAP50={res_new.box.map50:.4f}, mAP50-95={res_new.box.map:.4f}, P={res_new.box.mp:.4f}, R={res_new.box.mr:.4f}", flush=True)

except Exception as e:
    print('❌ Lỗi đánh giá YOLOv11 (mới):', e)

# Export kết quả ra JSON
with open('report/metrics_conf05.json', 'w', encoding='utf-8') as f:
    json.dump(metrics_report, f, indent=4)

print('\n✅ Đánh giá hoàn tất! Kết quả đã được lưu tại report/metrics_conf05.json', flush=True)
