import pydicom
import numpy as np
import cv2
from PIL import Image
import os

def load_dicom_as_image(filepath, target_size=(512, 512)):
    """
    Đọc file DICOM (.dcm) và chuyển đổi thành ảnh PIL Image để hiển thị trên GUI
    :param filepath: Đường dẫn tới file DICOM
    :param target_size: Kích thước muốn resize
    :return: Đối tượng PIL.Image (dễ dàng dùng trong tkinter/customtkinter)
    """
    try:
        # 1. Đọc dữ liệu Dicom
        dicom_data = pydicom.dcmread(filepath)
        pixel_array = dicom_data.pixel_array
        
        # 2. Rescale & Normalize về khoảng 0-255 để làm ảnh 8-bit
        # DICOM y tế thường có giá trị pixel rất rộng (Ví dụ Hounsfield units từ -1000 đến 3000)
        # Ta cần chuẩn hóa chênh lệch này để thấy rõ được phần mềm
        image_2d = pixel_array.astype(float)
        
        # Tùy chọn: Ở đây ta normalize toàn bộ dải, thực tế nếu lấy riêng phổi thì dùng Windowing (Window Center / Width)
        # Nhưng để hiển thị cơ bản, ta dùng min-max scaling
        image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
        
        # 3. Chuyển đổi thành uint8
        image_uint8 = np.uint8(image_2d_scaled)
        
        # 4. Chuyển mảng NumPy sang ảnh PIL
        # DICOM là ảnh grayscale
        img = Image.fromarray(image_uint8, mode='L')
        
        # 5. Resize ảnh
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        return img
    except Exception as e:
        print(f"Lỗi khi đọc file DICOM: {e}")
        return None

def load_dicom_series(directory_path, target_size=(512, 512)):
    """
    Đọc toàn bộ file DICOM trong một thư mục, sắp xếp theo Slice/Instance Number 
    và chuyển đổi thành danh sách ảnh PIL.
    """
    try:
        dicom_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith('.dcm')]
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
            pixel_array = ds.pixel_array
            image_2d = pixel_array.astype(float)
            # Windowing an toàn
            if image_2d.max() > 0:
                image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
            else:
                image_2d_scaled = image_2d
            image_uint8 = np.uint8(image_2d_scaled)
            img = Image.fromarray(image_uint8, mode='L')
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            pil_images.append(img)
            
        return pil_images
    except Exception as e:
        print(f"Lỗi khi đọc thư mục DICOM: {e}")
        return []
