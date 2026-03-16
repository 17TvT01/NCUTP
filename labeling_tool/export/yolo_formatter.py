from pathlib import Path
from data.xml_io import load_annotations_xml

def convert_to_yolo_format(annotations_xml: Path, output_dir: Path, image_width: int = 512, image_height: int = 512) -> None:
    data = load_annotations_xml(annotations_xml)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_mapping = {"nodule": 0, "suspicious": 1, "non-nodule": 2}
    for slice_idx_str, labels in data.get("annotations", {}).items():
        slice_idx = int(slice_idx_str)
        output_file = output_dir / f"slice_{slice_idx:04d}.txt"
        with open(output_file, "w") as f:
            for label in labels:
                bbox = label.get("bbox")
                if not bbox: continue
                label_type = label.get("label_type", "nodule")
                class_id = class_mapping.get(label_type, 0)
                x_center = (bbox[0] + bbox[2] / 2) / image_width
                y_center = (bbox[1] + bbox[3] / 2) / image_height
                width = bbox[2] / image_width
                height = bbox[3] / image_height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    print(f"YOLO format labels saved to {output_dir}")
