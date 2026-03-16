import numpy as np

def cluster_nodules_3d(all_slice_results, dist_threshold=20, max_slice_gap=3, min_slices=3):
    """
    Thuật toán ghép các Dấu vết 2D (trên nhiều lát cắt) thành 1 Khối Nốt Phổi 3D duy nhất.
    all_slice_results: dict { slice_idx: { "nodules": [n1, ...], ... } }
    Return: Danh sách các cụm (Clusters)
    """
    clusters = [] # Chứa các list của (slice_idx, nodule)
    
    # Duyệt tuần tự từ trên xuống dưới theo trục Z
    for s_idx in sorted(all_slice_results.keys()):
        nodules_in_slice = all_slice_results[s_idx].get("nodules", [])
        for n in nodules_in_slice:
            best_cluster_idx = -1
            best_iou = 0
            
            x1, y1, x2, y2 = n['x1'], n['y1'], n['x2'], n['y2']
            area1 = (x2 - x1) * (y2 - y1)
            
            # Tìm xem nốt này có đè lên nốt cũ nào ở lát cắt ngay sát trên không
            for c_idx, cluster in enumerate(clusters):
                last_s_idx, last_n = cluster[-1]
                
                # Nếu cách nhau quá xa (VD: nhảy cóc 4 lát cắt) thì chắc chắn không chung 1 nốt
                if s_idx - last_s_idx > max_slice_gap:
                    continue
                    
                # Tính Toán Độ Trượt Tâm (Centroid Distance)
                dist = np.sqrt((n['center_x'] - last_n['center_x'])**2 + (n['center_y'] - last_n['center_y'])**2)
                
                # Tính Toán Độ Đè (IoU)
                lx1, ly1, lx2, ly2 = last_n['x1'], last_n['y1'], last_n['x2'], last_n['y2']
                inter_x1, inter_y1 = max(x1, lx1), max(y1, ly1)
                inter_x2, inter_y2 = min(x2, lx2), min(y2, ly2)
                inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
                
                inter_area = inter_w * inter_h
                area2 = (lx2 - lx1) * (ly2 - ly1)
                iou = inter_area / float(area1 + area2 - inter_area + 1e-6)
                
                # Luật Nối Liền: Nếu tâm 2 box cách nhau dưới 20 Pixel, HOẶC diện tích đè > 20%
                if iou > 0.2 or dist < dist_threshold:
                    if iou > best_iou or dist < dist_threshold:
                        best_iou = max(iou, best_iou)
                        best_cluster_idx = c_idx
            
            # Gắn vào cụm có sẵn, nếu không thì tự tạo khối 3D mới
            if best_cluster_idx != -1:
                clusters[best_cluster_idx].append((s_idx, n))
            else:
                clusters.append([(s_idx, n)])
                
    # Tổng hợp danh sách Nốt 3D cuối cùng
    final_clusters = []
    for c in clusters:
        slices = [item[0] for item in c]
        
        # BỎ QUA NẾU NỐT QUÁ MỎNG (Ít hơn min_slices lát cắt)
        if len(slices) < min_slices:
            continue
        
        # Tìm lát cắt đại diện (Lát mà Nốt phình to nhất = Bụng khối U)
        best_slice_idx = -1
        max_area = 0
        best_n = None
        max_fpr = 0
        
        for s_idx, n in c:
            a = (n['x2'] - n['x1']) * (n['y2'] - n['y1'])
            if a > max_area:
                max_area = a
                best_slice_idx = s_idx
                best_n = n
                
            fpr = n.get('fpr_score', 1.0)
            if fpr > max_fpr:
                max_fpr = fpr # Lấy cảnh báo Ác tính cao nhất của Khối này
                
        cluster_info = {
            "id": len(final_clusters) + 1,
            "z_start": min(slices),
            "z_end": max(slices),
            "z_core": best_slice_idx, # Lõi trung tâm
            "total_slices": len(slices),
            "center_x": best_n['center_x'],
            "center_y": best_n['center_y'],
            "voxel": best_n.get('voxel', best_n.get('morph_area', max_area)),
            "fpr_score": max_fpr,
            "x1": best_n['x1'], "y1": best_n['y1'],
            "x2": best_n['x2'], "y2": best_n['y2']
        }
        final_clusters.append(cluster_info)
        
    return final_clusters
