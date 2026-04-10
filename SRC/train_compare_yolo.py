import os
import argparse
import torch
from ultralytics import YOLO

def train_model(version, data_yaml, epochs, batch_size, imgsz):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN YOLO{version.upper()} TRÊN THIẾT BỊ: {device.upper()}")
    print(f"   Epochs: {epochs} | Batch: {batch_size} | Image Size: {imgsz}")
    print(f"{'='*60}")
    
    # Xác định kiến trúc mô hình
    if version == 'v8':
        model_name = "yolov8n.pt"
    elif version == 'v11':
        model_name = "yolo11n.pt"  # Ultralytics bỏ chữ 'v' từ phiên bản 11
    else:
        print("❌ Phiên bản không hợp lệ! Vui lòng chọn 'v8' hoặc 'v11'")
        return

    # Khởi tạo mô hình pre-trained mặc định từ Ultralytics (tải về nếu chưa có)
    print(f"⚙️ Khởi tạo mô hình kiến trúc: {model_name}...")
    model = YOLO(model_name)
    
    # Tiến hành Huấn luyện (Chạy trên Server với cấu hình cực đoan để bắt nốt nhỏ)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=os.path.abspath('runs_compare'), # Tách riêng ra thư mục runs_compare
        name=f'train_yolo{version}',
        plots=True,          # Tạo biểu đồ để tiện so sánh
        verbose=True,
        workers=8,           # Đa luồng Server để load ảnh

        # --- CÁC THAM SỐ GÂY ĐỘT BIẾN Y TẾ ---
        mosaic=1.0,  
        mixup=0.1,
        box=10.0,
        cls=2.0,
        fliplr=0.0,  # Không lật ngang đối với Phổi
        flipud=0.0   # Không lật dọc
    )
    
    print(f"\n✅ Hoàn tất huấn luyện YOLO{version}.")
    best_weight = os.path.join(os.path.abspath('runs_compare'), f'train_yolo{version}', 'weights', 'best.pt')
    print(f"🎯 Trọng số tốt nhất đã được lưu tại:\n👉 {best_weight}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Công cụ huấn luyện YOLO độc lập (Hỗ trợ cấu hình chạy Server)")
    parser.add_argument('--version', type=str, choices=['v8', 'v11', 'both'], default='both', 
                        help="Chọn phiên bản YOLO để huấn luyện (v8, v11, hoặc both để chạy tuần tự)")
    # Mặc định đường dẫn file YAML đang nằm trong thư mục dataset_yolo_final theo như cấu trúc của bạn
    parser.add_argument('--data', type=str, default='dataset_yolo_final/data.yaml', 
                        help="Đường dẫn tương đối/tuyệt đối đến file cấu hình dataset (.yaml)")
    parser.add_argument('--epochs', type=int, default=100, help="Số lượng epochs để huấn luyện (Nên để 300 nếu chạy Server)")
    parser.add_argument('--batch', type=int, default=8, help="Kích thước batch size (Nên dùng 16 hoặc 32 cho Server)")
    parser.add_argument('--imgsz', type=int, default=640, help="Kích thước ảnh đầu vào (Khuyến nghị 1024 trên Server)")

    args = parser.parse_args()

    # Xử lý đường dẫn file yaml tuyệt đối (Ultralytics yêu cầu đường dẫn rõ ràng)
    data_yaml = os.path.abspath(args.data)
    if not os.path.exists(data_yaml):
         print(f"\n❌ LỖI NGHIÊM TRỌNG: Lỗi không tìm thấy file dataset!")
         print(f"Đường dẫn đã tìm: {data_yaml}")
         print("Vui lòng kiểm tra lại xem thư mục 'dataset_yolo_final' hoặc tham số '--data' đã đúng tên chưa.")
         exit(1)

    print("\nBẮT ĐẦU CHƯƠNG TRÌNH ĐÁNH GIÁ & HUẤN LUYỆN YOLO")
    print(f"📁 Dataset Path: {data_yaml}")

    # Chạy huấn luyện dựa trên tham số lựa chọn
    if args.version in ['v8', 'both']:
        train_model('v8', data_yaml, args.epochs, args.batch, args.imgsz)
        
    if args.version in ['v11', 'both']:
        train_model('v11', data_yaml, args.epochs, args.batch, args.imgsz)

    print("\n🎉 ĐÃ HOÀN THÀNH TOÀN BỘ CÁC TIẾN TRÌNH HUẤN LUYỆN SO SÁNH!")
    print("💡 Mẹo: Bạn hãy mở thư mục 'runs_compare' để xem các file 'results.png', 'confusion_matrix.png' của cả 2 model để có cái nhìn tổng quan nhất.")
