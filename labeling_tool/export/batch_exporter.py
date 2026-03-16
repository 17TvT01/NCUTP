import cv2
import numpy as np
from pathlib import Path
import shutil
import random
from data.dicom_loader import load_dicom_volume
from data.xml_io import load_annotations_xml

def create_dataset_yaml(output_dir: Path, dataset_name: str = "nodule_detection") -> None:
    yaml_content = f"""# {dataset_name} Dataset Configuration\npath: {output_dir.absolute()}\ntrain: images\nval: images\nnames:\n  0: nodule\n  1: suspicious\n  2: non-nodule\nnc: 3\n"""
    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)
    print(f"Created dataset configuration: {output_dir / 'dataset.yaml'}")

def export_labeled_slices(annotations_xml: Path, dicom_dir: Path, output_dir: Path) -> None:
    data = load_annotations_xml(annotations_xml)
    volume = load_dicom_volume(list(dicom_dir.glob("*.dcm")))
    images_dir, labels_dir = output_dir / "images", output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    class_mapping = {"nodule": 0, "suspicious": 1, "non-nodule": 2}
    for slice_idx_str, labels in data.get("annotations", {}).items():
        slice_idx = int(slice_idx_str)
        if slice_idx >= len(volume): continue
        slice_img = volume[slice_idx]
        cv2.imwrite(str(images_dir / f"slice_{slice_idx:04d}.png"), (slice_img * 255).astype(np.uint8))
        
        with open(labels_dir / f"slice_{slice_idx:04d}.txt", "w") as f:
            for label in labels:
                bbox = label.get("bbox")
                if not bbox: continue
                class_id = class_mapping.get(label.get("label_type", "nodule"), 0)
                h, w = slice_img.shape
                xc, yc = (bbox[0] + bbox[2] / 2) / w, (bbox[1] + bbox[3] / 2) / h
                nw, nh = bbox[2] / w, bbox[3] / h
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
    print(f"Exported {len(data['annotations'])} labeled slices to {output_dir}")

def split_dataset(images_dir: Path, labels_dir: Path, output_dir: Path, train_ratio: float = 0.8, val_ratio: float = 0.1) -> None:
    image_files = sorted(list(images_dir.glob("*.png")))
    random.shuffle(image_files)
    n_total = len(image_files)
    n_train, n_val = int(n_total * train_ratio), int(n_total * val_ratio)
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
    def copy_split(files, split_name):
        for img_file in files:
            shutil.copy(img_file, output_dir / split_name / "images" / img_file.name)
            label_file = labels_dir / img_file.with_suffix(".txt").name
            if label_file.exists():
                shutil.copy(label_file, output_dir / split_name / "labels" / label_file.name)

    copy_split(train_files, "train")
    copy_split(val_files, "val")
    copy_split(test_files, "test")
    print(f"Dataset split complete: Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
