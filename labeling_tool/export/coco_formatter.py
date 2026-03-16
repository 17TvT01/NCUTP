import json
from pathlib import Path
from data.xml_io import load_annotations_xml

def convert_to_coco_format(annotations_xml: Path, output_json: Path) -> None:
    data = load_annotations_xml(annotations_xml)
    coco = {"images": [], "annotations": [], "categories": [{"id": 0, "name": "nodule"}, {"id": 1, "name": "suspicious"}, {"id": 2, "name": "non-nodule"}]}
    class_mapping = {"nodule": 0, "suspicious": 1, "non-nodule": 2}
    annotation_id = 0
    for slice_idx_str, labels in data.get("annotations", {}).items():
        slice_idx = int(slice_idx_str)
        coco["images"].append({"id": slice_idx, "file_name": f"slice_{slice_idx:04d}.png", "width": 512, "height": 512})
        for label in labels:
            bbox = label.get("bbox")
            if not bbox: continue
            label_type = label.get("label_type", "nodule")
            category_id = class_mapping.get(label_type, 0)
            coco["annotations"].append({
                "id": annotation_id, "image_id": slice_idx, "category_id": category_id,
                "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0
            })
            annotation_id += 1
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"COCO format saved to {output_json}")
