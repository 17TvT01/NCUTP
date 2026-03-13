import os
import sys
# Cấp quyền import thư mục cha
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import pydicom
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk
from scipy.ndimage import binary_fill_holes

from utils.image_reader import _dicom_to_hu, _apply_lung_window

def generate_lung_mask(hu_image):
    """
    Sử dụng thuật toán Xử lý ảnh truyền thống để mò ra 2 lá phổi từ ảnh CT (HU).
    Lung thường có giá trị HU < -400.
    """
    # 1. Ngưỡng (Thresholding) để lấy khí (trong phổi + ngoài cơ thể)
    binary = hu_image < -400
    
    # 2. Xóa các vùng dính rìa ảnh (Khí ngoài cơ thể)
    cleared = clear_border(binary)
    
    # 3. Label các vùng dính nhau
    label_image = label(cleared)
    
    # 4. Tìm 2 vùng to nhất (Thường là 2 lá phổi, loại bỏ bong bóng khí trong ruột)
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    
    # 5. Fill các lỗ hổng bên trong phổi (Mạch máu, Nốt phổi) để thành cục phổi đặc
    filled_lung = binary_fill_holes(binary)
    
    # 6. Smooth rìa bằng Closing
    selem = disk(3)
    final_mask = closing(filled_lung, selem)
    
    return np.uint8(final_mask * 255)

def process_directory(input_dir, output_img_dir, output_mask_dir):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    dicom_files = glob(os.path.join(input_dir, '*.dcm'))
    print(f"[{os.path.basename(input_dir)}] Tìm thấy {len(dicom_files)} lát cắt DICOM.")
    
    for f in tqdm(dicom_files):
        # Đọc Dicom
        try:
            ds = pydicom.dcmread(f)
            hu_image = _dicom_to_hu(ds)
            
            # Tạo Mask
            mask = generate_lung_mask(hu_image)
            
            # Nếu mảng mask hoàn toàn đen (Không tìm thấy phổi ở lát này), thì skip
            if np.sum(mask) == 0:
                continue
                
            # Tạo thư mục và save
            base_name = os.path.basename(f).replace('.dcm', '')
            
            # Lưu Mask
            cv2.imwrite(os.path.join(output_mask_dir, f"{base_name}_mask.png"), mask)
            
            # Mặc dù Data Train cần ghép với _apply_lung_window, nhưng hàm đó đang ở image_reader
            # Ta sẽ import nó
            from image_reader import _apply_lung_window
            img_windowed = _apply_lung_window(hu_image)
            cv2.imwrite(os.path.join(output_img_dir, f"{base_name}.png"), img_windowed)
            
        except Exception as e:
            print(f"Lỗi file {f}: {e}")

if __name__ == '__main__':
    # Các thư mục CT hiện có
    data_folders = [
        "d:/Tool-vibecode/NCS/data/3000566.000000-NA-03192",
        "d:/Tool-vibecode/NCS/data/3000571.000000-NA-93273",
        "d:/Tool-vibecode/NCS/data/ct nguc bn tran duy khanh"
    ]
    
    out_img = "d:/Tool-vibecode/NCS/dataset_unet/images"
    out_mask = "d:/Tool-vibecode/NCS/dataset_unet/masks"
    
    for folder in data_folders:
        if os.path.exists(folder):
            process_directory(folder, out_img, out_mask)
    
    print("XONG! Đã sinh xong dữ liệu huấn luyện UNet.")
