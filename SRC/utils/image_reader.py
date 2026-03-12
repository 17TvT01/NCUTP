import pydicom
import numpy as np
import cv2
from PIL import Image
import os

# === Hằng số Lung Window chuẩn Y khoa ===
# Center = -600 HU (Trung tâm mật độ nhu mô phổi)
# Width  = 1500 HU (Dải rộng để bắt cả nốt vôi hóa đến khí phế nang)
LUNG_WINDOW_CENTER = -600
LUNG_WINDOW_WIDTH  = 1500

def _dicom_to_hu(dicom_data):
    """Chuyển pixel_array thô từ DICOM sang đơn vị chuẩn Hounsfield (HU).
    Công thức: HU = pixel_value * RescaleSlope + RescaleIntercept
    """
    pixel_array = dicom_data.pixel_array.astype(np.float32)
    slope = float(getattr(dicom_data, 'RescaleSlope', 1.0))
    intercept = float(getattr(dicom_data, 'RescaleIntercept', 0.0))
    hu_image = pixel_array * slope + intercept
    return hu_image

def _apply_lung_window(hu_image, center=LUNG_WINDOW_CENTER, width=LUNG_WINDOW_WIDTH):
    """Áp dụng Lung Windowing: Cắt dải HU chỉ giữ vùng phổi.
    Mọi thứ ngoài dải [center - width/2, center + width/2] bị kẹp về 0 hoặc 255.
    """
    min_hu = center - width / 2  # -1350 HU
    max_hu = center + width / 2  # +150  HU
    
    # Kẹp giá trị vào dải cho phép
    hu_clipped = np.clip(hu_image, min_hu, max_hu)
    
    # Scale tuyến tính từ [min_hu, max_hu] -> [0, 255]
    hu_normalized = (hu_clipped - min_hu) / (max_hu - min_hu) * 255.0
    return np.uint8(hu_normalized)

def load_dicom_as_image(filepath, target_size=(512, 512)):
    """
    Đọc file DICOM (.dcm), chuẩn hóa HU + Lung Windowing, trả về ảnh PIL.
    """
    try:
        dicom_data = pydicom.dcmread(filepath)
        
        # Bước 1: Chuyển sang HU chuẩn quốc tế
        hu_image = _dicom_to_hu(dicom_data)
        
        # Bước 2: Áp dụng Lung Window -> Chỉ giữ lại mật độ phổi
        image_uint8 = _apply_lung_window(hu_image)
        
        # Bước 3: Chuyển sang PIL Image grayscale
        img = Image.fromarray(image_uint8, mode='L')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        return img
    except Exception as e:
        print(f"Lỗi khi đọc file DICOM: {e}")
        return None

def load_dicom_series(directory_path, target_size=(512, 512)):
    """
    Đọc toàn bộ file DICOM trong một thư mục, sắp xếp theo Instance Number,
    chuẩn hóa HU + Lung Windowing, trả về danh sách ảnh PIL.
    """
    try:
        dicom_files = [
            os.path.join(directory_path, f) 
            for f in os.listdir(directory_path) 
            if f.lower().endswith('.dcm')
        ]
        if not dicom_files:
            return []
        
        slices = []
        for f in dicom_files:
            try:
                ds = pydicom.dcmread(f)
                slices.append(ds)
            except:
                pass
                
        # Sắp xếp theo InstanceNumber để các lát cắt liên tục nhau
        slices.sort(key=lambda x: int(getattr(x, 'InstanceNumber', 0)))
        
        pil_images = []
        for ds in slices:
            # Bước 1: Chuyển sang HU chuẩn quốc tế
            hu_image = _dicom_to_hu(ds)
            
            # Bước 2: Áp dụng Lung Window
            image_uint8 = _apply_lung_window(hu_image)
            
            # Bước 3: Chuyển sang PIL Image
            img = Image.fromarray(image_uint8, mode='L')
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            pil_images.append(img)
            
        return pil_images
    except Exception as e:
        print(f"Lỗi khi đọc thư mục DICOM: {e}")
        return []
