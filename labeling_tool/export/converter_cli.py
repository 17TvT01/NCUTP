import argparse
from pathlib import Path
from export.yolo_formatter import convert_to_yolo_format
from export.coco_formatter import convert_to_coco_format
from export.patch_extractor import extract_patches
from export.batch_exporter import export_labeled_slices, split_dataset, create_dataset_yaml

def main_converter():
    parser = argparse.ArgumentParser(description="Convert XML labeled data to training format")
    parser.add_argument("--annotations", type=str, required=True, help="Path to annotations XML")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--format", type=str, choices=["yolo", "coco", "patches", "batch"], default="yolo", help="Output format")
    parser.add_argument("--dicom-dir", type=str, help="Path to DICOM for batch export")
    parser.add_argument("--split", action="store_true", help="Split dataset into train/val/test when using batch")
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    output_path = Path(args.output)

    if args.format == "yolo":
        convert_to_yolo_format(annotations_path, output_path / "labels")
    elif args.format == "coco":
        convert_to_coco_format(annotations_path, output_path / "annotations.json")
    elif args.format == "patches":
        print("For patches format, you need to provide the volume separately via python script.")
    elif args.format == "batch":
        if not args.dicom_dir:
            print("Error: --dicom-dir is required for batch export")
            return
        dicom_dir = Path(args.dicom_dir)
        temp_dir = output_path / "temp"
        export_labeled_slices(annotations_path, dicom_dir, temp_dir)
        if args.split:
            split_dataset(temp_dir / "images", temp_dir / "labels", output_path)
            create_dataset_yaml(output_path)
        else:
            create_dataset_yaml(temp_dir)
        print("Export complete!")

if __name__ == "__main__":
    main_converter()
