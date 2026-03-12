import os
import xml.etree.ElementTree as ET
import numpy as np
import pydicom
import cv2
import json

def parse_lidc_xml(xml_path):
    sop_to_boxes = {}
    try:
        tree = ET.parse(xml_path)
        for roi in tree.iter():
            if roi.tag.endswith('roi'):
                sop_uid = None
                x_coords = []
                y_coords = []
                for child in roi.iter():
                    if child.tag.endswith('imageSOP_UID'): sop_uid = child.text.strip()
                    elif child.tag.endswith('xCoord'): x_coords.append(float(child.text))
                    elif child.tag.endswith('yCoord'): y_coords.append(float(child.text))
                
                if sop_uid and x_coords and y_coords:
                    xmin, xmax = min(x_coords), max(x_coords)
                    ymin, ymax = min(y_coords), max(y_coords)
                    if sop_uid not in sop_to_boxes: sop_to_boxes[sop_uid] = []
                    sop_to_boxes[sop_uid].append([xmin, ymin, xmax, ymax])
    except Exception as e: print(f"Lỗi đọc LIDC XML: {e}")
    return sop_to_boxes

def load_dicom_volume(dicom_dir):
    """
    Đọc TOÀN BỘ thư mục DICOM thành một khối Voxel 3D Numpy Array.
    Sắp xếp thứ tự Z theo InstanceNumber.
    Trả về: (volume_3d_numpy, list_sop_uids)
    """
    slices = []
    files = [f for f in os.listdir(dicom_dir) if f.lower().endswith('.dcm')]
    for file in files:
        fpath = os.path.join(dicom_dir, file)
        try:
            ds = pydicom.dcmread(fpath)
            # Windowing Phổi
            img = ds.pixel_array
            if 'RescaleSlope' in ds: img = img * ds.RescaleSlope + ds.RescaleIntercept
            wc, ww = -600, 1500
            img = np.clip(img, wc - ww//2, wc + ww//2)
            img = (img - (wc - ww//2)) / ww * 255.0
            
            slices.append({
                'z': int(ds.InstanceNumber) if 'InstanceNumber' in ds else 0,
                'uid': ds.SOPInstanceUID,
                'pixel': img.astype(np.uint8)
            })
        except: pass
        
    slices.sort(key=lambda s: s['z'])
    
    volume = np.stack([s['pixel'] for s in slices]) # Shape: (D, H, W)
    uids = [s['uid'] for s in slices]
    return volume, uids

def extract_3d_patches(dicom_dir, xml_dir, out_file="dataset_3d.npz", patch_size=(16, 32, 32)):
    """
    Cắt Hộp Diêm: (Depth, Height, Width)
    Depth=16 (Trục Z), Height=32, Width=32
    """
    print(f"Đang gom dữ liệu 3D từ {dicom_dir}...")
    master_sop_map = {}
    for f in os.listdir(xml_dir):
        if f.lower().endswith('.xml'):
            master_sop_map.update(parse_lidc_xml(os.path.join(xml_dir, f)))
            
    volume, uids = load_dicom_volume(dicom_dir)
    D, H, W = volume.shape
    pD, pH, pW = patch_size
    
    patches = []
    labels = []
    
    # Duyệt qua từng lát cắt Z
    for z, uid in enumerate(uids):
        if uid in master_sop_map:
            # 1. CẮT NỐT THẬT (Label = 1)
            for box in master_sop_map[uid]:
                xmin, ymin, xmax, ymax = box
                cx, cy = int((xmin + xmax)//2), int((ymin + ymax)//2)
                
                # Tính bounding box Không gian 3 Chiều
                z_start = max(0, z - pD//2)
                z_end = min(D, z_start + pD)
                y_start = max(0, cy - pH//2)
                y_end = min(H, y_start + pH)
                x_start = max(0, cx - pW//2)
                x_end = min(W, x_start + pW)
                
                # Trích xuất Voxel 3D
                patch = volume[z_start:z_end, y_start:y_end, x_start:x_end]
                
                # Padding bằng mảng 0 nếu Voxel chạm viền màn hình (Bị hụt kích thước 16x32x32)
                if patch.shape != patch_size:
                    pad_z = (0, max(0, patch_size[0] - patch.shape[0]))
                    pad_y = (0, max(0, patch_size[1] - patch.shape[1]))
                    pad_x = (0, max(0, patch_size[2] - patch.shape[2]))
                    patch = np.pad(patch, (pad_z, pad_y, pad_x), mode='constant', constant_values=0)
                
                patches.append(patch)
                labels.append(1) # 1 = Nodule
                
                # 2. CẮT RÁC NHIỄU GIẢ (Label = 0)
                # Random 2 cái hộp ngẫu nhiên xung quanh quả phổi nhưng không được trùng Box Thật
                for _ in range(2):
                    rx = np.random.randint(pW//2, W - pW//2)
                    ry = np.random.randint(pH//2, H - pH//2)
                    # Điều kiện không chạm vào tâm nốt thật
                    if abs(rx - cx) > pW and abs(ry - cy) > pH:
                        r_patch = volume[z_start:z_end, ry-pH//2:ry+(pH-pH//2), rx-pW//2:rx+(pW-pW//2)]
                        if r_patch.shape == patch_size:
                            patches.append(r_patch)
                            labels.append(0) # 0 = Non-Nodule

    if patches:
        np.savez_compressed(out_file, x=np.array(patches), y=np.array(labels))
        print(f"✅ Hoàn tất! Đã tạo được {sum(labels)} khối Nốt Phổi Thật và {len(labels)-sum(labels)} khối Mạch Máu Giả.")
        print(f"📦 Khối dữ liệu 3D lưu tại: {out_file}")
    else:
        print("❌ Không tìm thấy tọa độ Nốt nào để cắt hộp 3D!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="Thư mục chứa file DICOM")
    parser.add_argument("--xml_dir", type=str, required=True, help="Thư mục chứa file phân vùng LIDC XML")
    parser.add_argument("--out", type=str, default="dataset_3d_nodules.npz", help="Tên file khối NumPy Output")
    args = parser.parse_args()
    
    extract_3d_patches(args.img_dir, args.xml_dir, args.out)
