import os
import shutil
import random
import xml.etree.ElementTree as ET
import cv2
import yaml
import numpy as np
import pydicom

try:
    import albumentations as A
except ImportError:
    print("Thư viện albumentations chưa được cài đặt.")
    A = None

def parse_lidc_xml(xml_path):
    """
    Phân tích file XML của bộ dữ liệu LIDC-IDRI.
    Trả về Dictionary: SOPInstanceUID -> Danh sách Bounding Box [xmin, ymin, xmax, ymax]
    """
    sop_to_boxes = {}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for roi in root.iter():
            if roi.tag.endswith('roi'):
                sop_uid = None
                x_coords = []
                y_coords = []
                for child in roi.iter():
                    if child.tag.endswith('imageSOP_UID'):
                        sop_uid = child.text.strip()
                    elif child.tag.endswith('xCoord'):
                        x_coords.append(float(child.text))
                    elif child.tag.endswith('yCoord'):
                        y_coords.append(float(child.text))
                
                if sop_uid and x_coords and y_coords:
                    xmin, xmax = min(x_coords), max(x_coords)
                    ymin, ymax = min(y_coords), max(y_coords)
                    if sop_uid not in sop_to_boxes:
                        sop_to_boxes[sop_uid] = []
                    # Cộng thêm 2 pixel padding cho nốt rộng ra tí
                    sop_to_boxes[sop_uid].append([xmin-2, ymin-2, xmax+2, ymax+2])
    except Exception as e:
        print(f"Lỗi đọc LIDC XML: {e}")
    return sop_to_boxes

def normalize_yolo_bbox(box, w, h):
    """ Normalize [xmin, ymin, xmax, ymax] sang YOLO [x_center, y_center, width, height] """
    xmin, ymin, xmax, ymax = box
    x_center = ((xmin + xmax) / 2) / w
    y_center = ((ymin + ymax) / 2) / h
    width = (xmax - xmin) / w
    height = (ymax - ymin) / h
    return [x_center, y_center, width, height]

def convert_dicom_to_cv2(dcm_path):
    """Đọc file DICOM và chuyển thành ảnh RGB numpy array chuẩn OpenCV"""
    ds = pydicom.dcmread(dcm_path)
    sop_uid = ds.SOPInstanceUID
    pixel_array = ds.pixel_array
    
    if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
        pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        
    # Áp dụng Windowing cứng cho phổi (W: 1500, L: -600)
    window_center = -600
    window_width = 1500
    
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    
    pixel_array = np.clip(pixel_array, img_min, img_max)
    pixel_array = (pixel_array - img_min) / window_width * 255.0
    
    img_8bit = pixel_array.astype(np.uint8)
    img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)
    return img_rgb, sop_uid

def create_dataset(image_dir, xml_dir, output_dir, classes=["nodule"], augment_factor=5, split_ratio=0.8):
    """
    Tự động chia tách dữ liệu và nhân bản ảnh theo định dạng LIDC-IDRI.
    """
    print(f"Bắt đầu xử lý dữ liệu từ [{image_dir}] và [{xml_dir}]")
    print(f"Thư mục Output: {output_dir}")
    
    # 1. Tạo cấu trúc thư mục YOLO
    folders = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for folder in folders:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
        
    # 2. Tìm tất cả file XML và parse gom lại 1 Dictionary tổng
    master_sop_map = {}
    valid_xml = [f for f in os.listdir(xml_dir) if f.lower().endswith('.xml')]
    for x_file in valid_xml:
        sop_map = parse_lidc_xml(os.path.join(xml_dir, x_file))
        master_sop_map.update(sop_map)
        
    if not master_sop_map:
        print("Không tìm thấy Box Nodule nào trong các file XML. Quá trình chuyển đổi thất bại.")
        return
        
    # 3. Lấy danh sách file DICOM
    valid_extensions = ('.dcm',)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print("Không tìm thấy file DICOM (.dcm) nào trong thư mục hình.")
        return
        
    random.shuffle(image_files)
    split_index = int(len(image_files) * split_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]
    
    aug_pipeline = None
    if A is not None and augment_factor > 1:
        aug_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5), 
            A.RandomBrightnessContrast(p=0.5), 
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5), 
            A.GaussNoise(p=0.2) 
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    processed_count = 0
    generated_count = 0
    
    def process_split(files, subset):
        nonlocal processed_count, generated_count
        for file in files:
            img_path = os.path.join(image_dir, file)
            # Tạo tên ngẫu nhiên độc nhất thay vì base_name để tránh trùng đè khi import nhiều folder
            random_id = hex(hash(file + image_dir))[-8:]
            base_name = f"{subset}_{random_id}_{os.path.splitext(file)[0]}"
            
            try:
                img, sop_uid = convert_dicom_to_cv2(img_path)
            except Exception as e:
                continue
                
            # Kiểm tra xem ID của lát cắt CT này có nằm trong danh sách Bệnh án XML báo cáo không?
            if sop_uid not in master_sop_map:
                continue
                
            bboxes = master_sop_map[sop_uid]
            class_labels = [0] * len(bboxes) # Class = 0 (Nodule)
            
            def save_sample(image, box_list, label_list, img_name, is_val=False):
                nonlocal generated_count
                out_img_dir = os.path.join(output_dir, 'images', 'val' if is_val else 'train')
                out_lbl_dir = os.path.join(output_dir, 'labels', 'val' if is_val else 'train')
                
                cv2.imwrite(os.path.join(out_img_dir, img_name + '.jpg'), image)
                
                with open(os.path.join(out_lbl_dir, img_name + '.txt'), 'w') as f:
                    for box, cls_id in zip(box_list, label_list):
                        yolo_box = normalize_yolo_bbox(box, image.shape[1], image.shape[0])
                        if all(0 <= val <= 1 for val in yolo_box):
                            f.write(f"{cls_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n")
                generated_count += 1

            # Lưu ảnh Gốc
            save_sample(img, bboxes, class_labels, base_name, is_val=(subset=='val'))
            processed_count += 1
            
            # Tăng cường dữ liệu
            if subset == 'train' and aug_pipeline is not None:
                for idx in range(1, augment_factor):
                    try:
                        augmented = aug_pipeline(image=img, bboxes=bboxes, class_labels=class_labels)
                        if len(augmented['bboxes']) > 0:
                            save_sample(augmented['image'], augmented['bboxes'], augmented['class_labels'], f"{base_name}_aug_{idx}")
                    except Exception as e:
                        pass

    process_split(train_files, 'train')
    process_split(val_files, 'val')
    
    # 4. Tạo file data.yaml
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print("-" * 40)
    print("HOÀN TẤT CHUYỂN ĐỔI VÀ TĂNG CƯỜNG DỮ LIỆU!")
    print(f"- Số ảnh gốc đã xử lý có Nodule: {processed_count}")
    print(f"- Tổng số mẫu dữ liệu YOLO tạo ra: {generated_count} (Gấp {generated_count/processed_count if processed_count>0 else 0:.1f} lần)")
    print(f"- File cấu hình Training: {yaml_path}")
    print("-> Sẵn sàng để đưa đường dẫn data.yaml vào ứng dụng!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert XML to YOLO and Augment Data")
    parser.add_argument("--img_dir", type=str, required=True, help="Đường dẫn thư mục chứa Ảnh PNG/JPG/DCM")
    parser.add_argument("--xml_dir", type=str, required=True, help="Đường dẫn thư mục chứa XML (nhãn)")
    parser.add_argument("--out_dir", type=str, default="dataset_yolo", help="Thư mục xuất kết quả")
    parser.add_argument("--classes", type=str, default="nodule", help="Tên các classes (phân cách bằng dấu phẩy)")
    parser.add_argument("--aug", type=int, default=5, help="Hệ số nhân bản dữ liệu (Ví dụ 5 = 1 ảnh gốc đẻ ra thêm 4 ảnh ảo)")
    
    args = parser.parse_args()
    class_list = [c.strip() for c in args.classes.split(",")]
    create_dataset(args.img_dir, args.xml_dir, args.out_dir, class_list, args.aug)
