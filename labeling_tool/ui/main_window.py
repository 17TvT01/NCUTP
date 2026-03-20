from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QFileDialog, QMessageBox, QShortcut, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence

from .left_panel import LeftPanel
from .image_viewer import ImageViewer
from data.label import NoduleLabel
from data.dicom_loader import load_dicom_volume
from data.xml_io import save_annotations_xml, load_annotations_xml

class LabelingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nodule Labeling Tool (XML Edition)")
        self.setGeometry(100, 100, 1200, 800)

        self.dicom_files: List[Path] = []
        self.current_slice_idx = 0
        self.volume: Optional[np.ndarray] = None
        self.labels: Dict[int, List[NoduleLabel]] = {}
        self.current_label_type = "nodule"
        self.current_series_path: Optional[Path] = None

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()

        self.left_panel = LeftPanel()
        self.left_panel.load_dicom_sig.connect(self.load_dicom_series)
        self.left_panel.slice_changed_sig.connect(self.on_slice_changed)
        self.left_panel.label_type_changed_sig.connect(self.on_label_type_changed)
        self.left_panel.draw_mode_changed_sig.connect(self.set_draw_mode)
        self.left_panel.delete_annotation_sig.connect(self.delete_annotation)
        self.left_panel.clear_slice_sig.connect(self.clear_slice)
        self.left_panel.save_annotations_sig.connect(self.save_annotations)
        self.left_panel.load_annotations_sig.connect(self.load_annotations)
        self.left_panel.goto_marked_slice_sig.connect(self.go_to_selected_marker_slice)
        
        self.image_viewer = ImageViewer()
        self.image_viewer.annotations_changed_sig.connect(self.on_viewer_annotations_changed)
        self.image_viewer.setAlignment(Qt.AlignCenter)
        self.image_viewer.setMinimumSize(400, 400)
        self.image_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        main_layout.addWidget(self.left_panel, 1)
        main_layout.addWidget(self.image_viewer, 3)
        central_widget.setLayout(main_layout)

        QShortcut(QKeySequence("Ctrl+S"), self, self.save_annotations)
        QShortcut(QKeySequence("Ctrl+O"), self, self.load_dicom_series)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo_last_annotation)
        QShortcut(QKeySequence(Qt.Key_Delete), self, self.delete_annotation)

        # Keep references and force app-wide context so arrow keys work regardless of focused widget.
        self.prev_slice_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.prev_slice_shortcut.setContext(Qt.ApplicationShortcut)
        self.prev_slice_shortcut.activated.connect(self.prev_slice)

        self.next_slice_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.next_slice_shortcut.setContext(Qt.ApplicationShortcut)
        self.next_slice_shortcut.activated.connect(self.next_slice)

    def load_dicom_series(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Series Folder")
        if not folder: return
        folder_path = Path(folder)
        self.dicom_files = sorted(list(folder_path.glob("*.dcm")))
        if not self.dicom_files:
            QMessageBox.warning(self, "Error", "No DICOM files found")
            return

        try:
            self.current_series_path = folder_path
            self.volume = load_dicom_volume(self.dicom_files)
            self.labels = {}
            self.current_slice_idx = 0
            self.left_panel.update_info(folder_path.name, len(self.dicom_files))
            self.display_current_slice()
            self.update_stats()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load DICOM: {str(e)}")

    def display_current_slice(self):
        if self.volume is None: return
        self.image_viewer.set_image(self.volume[self.current_slice_idx])
        curr_anns = [l.to_dict() for l in self.labels.get(self.current_slice_idx, [])]
        self.image_viewer.set_annotations(curr_anns)
        self.left_panel.slice_label.setText(f"Slice: {self.current_slice_idx + 1}/{len(self.dicom_files)}")
        self.update_annotation_list()
        self.update_marked_slice_list()

    def on_slice_changed(self, value: int):
        self.save_current_slice_annotations()
        self.current_slice_idx = value
        self.display_current_slice()

    def save_current_slice_annotations(self):
        if not self.image_viewer.annotations:
            if self.current_slice_idx in self.labels:
                del self.labels[self.current_slice_idx]
                self.update_marked_slice_list()
                self.update_stats()
            return

        labels = []
        for ann in self.image_viewer.annotations:
            ann["slice_idx"] = self.current_slice_idx
            ann["label_type"] = self.current_label_type
            labels.append(NoduleLabel.from_dict(ann))
        self.labels[self.current_slice_idx] = labels
        self.update_marked_slice_list()
        self.update_stats()

    def on_viewer_annotations_changed(self):
        self.save_current_slice_annotations()
        self.update_annotation_list()

    def update_marked_slice_list(self):
        marker_counts = {slice_idx: len(lbls) for slice_idx, lbls in self.labels.items() if lbls}
        self.left_panel.update_marked_slice_list(marker_counts, self.current_slice_idx)

    def go_to_selected_marker_slice(self):
        target_slice_idx = self.left_panel.get_selected_marked_slice_index()
        if target_slice_idx < 0:
            QMessageBox.information(self, "Info", "Please select a marked slice first")
            return
        self.save_current_slice_annotations()
        self.left_panel.slice_slider.setValue(target_slice_idx)

    def set_draw_mode(self, mode: str):
        self.image_viewer.set_draw_mode(mode)
        instructions = {
            "rectangle": "Click and drag to draw rectangle",
            "circle": "Click center, drag to set radius",
            "ellipse": "Click and drag to draw ellipse",
            "polygon": "Click to add points, double/right click to finish",
            "freehand": "Click and drag to draw freely"
        }
        self.left_panel.set_instructions(instructions.get(mode, ""))

    def on_label_type_changed(self, label_type: str):
        self.current_label_type = label_type

    def update_annotation_list(self):
        self.left_panel.annotation_list.clear()
        for i, lbl in enumerate(self.labels.get(self.current_slice_idx, [])):
            shape = lbl.shape
            desc = ""
            if shape == "rectangle" and lbl.bbox:
                desc = f"Box: ({lbl.bbox[0]},{lbl.bbox[1]}) {lbl.bbox[2]}x{lbl.bbox[3]}"
            elif shape == "circle" and lbl.circle:
                desc = f"Circle: center({lbl.circle[0]},{lbl.circle[1]}) r={lbl.circle[2]}"
            elif shape == "ellipse" and lbl.bbox:
                desc = f"Ellipse: ({lbl.bbox[0]},{lbl.bbox[1]}) {lbl.bbox[2]}x{lbl.bbox[3]}"
            elif shape in ["polygon", "freehand"] and lbl.points:
                desc = f"{shape.capitalize()}: {len(lbl.points)} points"
            self.left_panel.annotation_list.addItem(f"{i + 1}. {lbl.label_type} - {desc}")

    def delete_annotation(self):
        idx = self.left_panel.get_selected_annotation_index()
        if idx >= 0 and self.current_slice_idx in self.labels:
            del self.labels[self.current_slice_idx][idx]
            if not self.labels[self.current_slice_idx]:
                del self.labels[self.current_slice_idx]
            self.image_viewer.annotations = [l.to_dict() for l in self.labels.get(self.current_slice_idx, [])]
            self.image_viewer.update_display()
            self.update_annotation_list()
            self.update_stats()

    def clear_slice(self):
        if self.current_slice_idx in self.labels:
            del self.labels[self.current_slice_idx]
            self.image_viewer.annotations = []
            self.image_viewer.update_display()
            self.update_annotation_list()
            self.update_marked_slice_list()
            self.update_stats()

    def undo_last_annotation(self):
        if self.image_viewer.annotations:
            self.image_viewer.annotations.pop()
            self.image_viewer.update_display()
            self.save_current_slice_annotations()
            self.update_annotation_list()

    def prev_slice(self):
        if self.current_slice_idx > 0:
            self.left_panel.slice_slider.setValue(self.current_slice_idx - 1)

    def next_slice(self):
        if self.current_slice_idx < len(self.dicom_files) - 1:
            self.left_panel.slice_slider.setValue(self.current_slice_idx + 1)

    def update_stats(self):
        total = sum(len(lbls) for lbls in self.labels.values())
        self.left_panel.stats_label.setText(f"Total annotations: {total}\nSlices with labels: {len(self.labels)}")

    def save_annotations(self):
        if not self.current_series_path:
            QMessageBox.warning(self, "Warning", "No series loaded")
            return
        self.save_current_slice_annotations()
        default_name = self.current_series_path.name + "_annotations.xml"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Annotations", str(self.current_series_path / default_name), "XML Files (*.xml)")
        if not file_path: return
        try:
            data = {
                "series_path": str(self.current_series_path),
                "num_slices": len(self.dicom_files),
                "annotations": {str(k): [l.to_dict() for l in v] for k, v in self.labels.items()}
            }
            save_annotations_xml(data, Path(file_path))
            QMessageBox.information(self, "Success", f"Saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")

    def load_annotations(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Annotations", "", "XML Files (*.xml)")
        if not file_path: return
        try:
            data = load_annotations_xml(Path(file_path))
            self.labels = {}
            for slice_idx_str, labels_data in data["annotations"].items():
                self.labels[int(slice_idx_str)] = [NoduleLabel.from_dict(ld) for ld in labels_data]
            self.display_current_slice()
            self.update_stats()
            QMessageBox.information(self, "Success", "Loaded successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {str(e)}")
