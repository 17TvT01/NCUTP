from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox, QButtonGroup, QRadioButton, QListWidget
from PyQt5.QtCore import Qt, pyqtSignal

class LeftPanel(QWidget):
    load_dicom_sig = pyqtSignal()
    slice_changed_sig = pyqtSignal(int)
    label_type_changed_sig = pyqtSignal(str)
    draw_mode_changed_sig = pyqtSignal(str)
    delete_annotation_sig = pyqtSignal()
    clear_slice_sig = pyqtSignal()
    save_annotations_sig = pyqtSignal()
    load_annotations_sig = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.load_btn = QPushButton("Load DICOM Series")
        self.load_btn.clicked.connect(self.load_dicom_sig.emit)
        layout.addWidget(self.load_btn)

        self.info_label = QLabel("No series loaded")
        layout.addWidget(self.info_label)

        self.slice_label = QLabel("Slice: 0/0")
        layout.addWidget(self.slice_label)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self.slice_changed_sig.emit)
        layout.addWidget(self.slice_slider)

        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Label Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["nodule", "non-nodule", "suspicious"])
        self.type_combo.currentTextChanged.connect(self.label_type_changed_sig.emit)
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)

        layout.addWidget(QLabel("Drawing Tool:"))
        self.tool_button_group = QButtonGroup()
        self._add_tool_btn(layout, "🔲 Rectangle", "rectangle", checked=True)
        self._add_tool_btn(layout, "⭕ Circle", "circle")
        self._add_tool_btn(layout, "⬭ Ellipse", "ellipse")
        self._add_tool_btn(layout, "📐 Polygon", "polygon")
        self._add_tool_btn(layout, "✏️ Freehand", "freehand")

        self.tool_instruction = QLabel("Click and drag to draw rectangle")
        self.tool_instruction.setWordWrap(True)
        layout.addWidget(self.tool_instruction)

        layout.addWidget(QLabel("Annotations on current slice:"))
        self.annotation_list = QListWidget()
        layout.addWidget(self.annotation_list)

        self.delete_btn = QPushButton("Delete Selected Annotation")
        self.delete_btn.clicked.connect(self.delete_annotation_sig.emit)
        layout.addWidget(self.delete_btn)

        self.clear_btn = QPushButton("Clear Current Slice")
        self.clear_btn.clicked.connect(self.clear_slice_sig.emit)
        layout.addWidget(self.clear_btn)

        self.save_btn = QPushButton("Save Annotations")
        self.save_btn.clicked.connect(self.save_annotations_sig.emit)
        layout.addWidget(self.save_btn)

        self.load_ann_btn = QPushButton("Load Annotations")
        self.load_ann_btn.clicked.connect(self.load_annotations_sig.emit)
        layout.addWidget(self.load_ann_btn)

        self.stats_label = QLabel("Total annotations: 0")
        layout.addWidget(self.stats_label)

    def _add_tool_btn(self, layout, text, mode, checked=False):
        btn = QRadioButton(text)
        btn.setChecked(checked)
        btn.toggled.connect(lambda state, m=mode: self.draw_mode_changed_sig.emit(m) if state else None)
        self.tool_button_group.addButton(btn)
        layout.addWidget(btn)

    def update_info(self, folder_name: str, num_slices: int):
        self.info_label.setText(f"Loaded: {folder_name}\nSlices: {num_slices}")
        self.slice_slider.setMaximum(max(0, num_slices - 1))
        self.slice_slider.setValue(0)

    def get_selected_annotation_index(self) -> int:
        return self.annotation_list.currentRow()
        
    def set_instructions(self, text: str):
        self.tool_instruction.setText(text)
