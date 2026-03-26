from ultralytics import YOLO

print("Evaluating YOLOv8...")
model_v8 = YOLO('best_yolo_medical.pt')
model_v8.val(data='dataset_yolo_final/data.yaml', plots=True, project='runs_compare', name='eval_v8', exist_ok=True, workers=0)

print("Evaluating YOLOv11...")
model_v11 = YOLO('runs_compare/train_yolov11/weights/best.pt')
model_v11.val(data='dataset_yolo_final/data.yaml', plots=True, project='runs_compare', name='eval_v11', exist_ok=True, workers=0)
