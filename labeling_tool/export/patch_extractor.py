import numpy as np
from pathlib import Path
from typing import Tuple
from data.xml_io import load_annotations_xml

def extract_patches(annotations_xml: Path, volume: np.ndarray, output_dir: Path, patch_size: Tuple[int, int] = (64, 64)) -> None:
    data = load_annotations_xml(annotations_xml)
    output_dir.mkdir(parents=True, exist_ok=True)
    pos_dir = output_dir / "positive"; pos_dir.mkdir(exist_ok=True)
    neg_dir = output_dir / "negative"; neg_dir.mkdir(exist_ok=True)
    sus_dir = output_dir / "suspicious"; sus_dir.mkdir(exist_ok=True)
    patch_h, patch_w = patch_size

    for slice_idx_str, labels in data.get("annotations", {}).items():
        slice_idx = int(slice_idx_str)
        if slice_idx >= len(volume): continue
        slice_img = volume[slice_idx]

        for i, label in enumerate(labels):
            bbox = label.get("bbox")
            if not bbox: continue
            label_type = label.get("label_type", "nodule")
            cx, cy = int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)
            y1, y2 = max(0, cy - patch_h // 2), min(slice_img.shape[0], cy + patch_h // 2)
            x1, x2 = max(0, cx - patch_w // 2), min(slice_img.shape[1], cx + patch_w // 2)
            patch = slice_img[y1:y2, x1:x2]
            if patch.shape != patch_size:
                padded = np.zeros(patch_size, dtype=patch.dtype)
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded
                
            save_dir = pos_dir if label_type == "nodule" else (sus_dir if label_type == "suspicious" else neg_dir)
            np.save(save_dir / f"slice_{slice_idx:04d}_bbox_{i:02d}.npy", patch)
    print(f"Patches extracted to {output_dir}")
