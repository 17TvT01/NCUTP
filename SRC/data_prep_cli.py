import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.data_prep import create_dataset

def run_batch_cli(input_parent_dir, out_dir, aug_val):
    if not os.path.exists(input_parent_dir):
        print(f"❌ Không tìm thấy thư mục cha: {input_parent_dir}")
        print("Vui lòng tạo thư mục này và bỏ 10 folder DICOM+XML của bạn vào trong đó.")
        sys.exit(1)

    folders = [os.path.join(input_parent_dir, d) for d in os.listdir(input_parent_dir) 
               if os.path.isdir(os.path.join(input_parent_dir, d))]
    
    if not folders:
        print(f"❌ Không có thư mục con nào bên trong {input_parent_dir}!")
        sys.exit(1)
        
    print(f"🔍 Đã quét thấy {len(folders)} thư mục mẫu.")
    print("=" * 50)
    
    for idx, folder in enumerate(folders):
        print(f"\n🚀 Đang xử lý [{idx+1}/{len(folders)}]: {folder}")
        
        create_dataset(
            image_dir=folder,
            xml_dir=folder,
            output_dir=out_dir,
            classes=["nodule"],
            augment_factor=aug_val
        )
        
    print("\n✅ HOÀN TẤT SINH DỮ LIỆU BATCH TRÊN SERVER UBUNTU!")
    print(f"Tất cả file đã được tổng hợp tại: {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy sinh dữ liệu hàng loạt không cần Giao diện (dành cho Server Ubuntu)")
    parser.add_argument("--input", type=str, default="data_samples", help="Thư mục cha chứa 10 folder DICOM")
    parser.add_argument("--output", type=str, default="dataset_yolo_final", help="Thư mục xuất dữ liệu YOLO")
    parser.add_argument("--aug", type=int, default=5, help="Hệ số nhân bản dữ liệu (Data Augmentation)")
    
    args = parser.parse_args()
    
    print("\n--- TRÌNH SINH DỮ LIỆU HEADLESS (DÀNH CHO SERVER) ---")
    run_batch_cli(args.input, args.output, args.aug)
