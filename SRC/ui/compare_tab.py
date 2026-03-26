import customtkinter as ctk
from PIL import Image
import numpy as np
import cv2
from ui.result_tree import ResultTree


class CompareTab(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.raw_images = []
        self.analysis_results = {}
        self.clusters = []
        self.fill_enabled = True
        self.current_idx = 0
        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.lbl_status = ctk.CTkLabel(
            self,
            text="Tab này dùng ảnh gốc + kết quả khoanh từ tab Phân tích.",
            anchor="w",
        )
        self.lbl_status.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=3)
        content.rowconfigure(0, weight=1)

        self.result_tree = ResultTree(content, on_item_click_cb=self.go_to_slice)
        self.result_tree.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        viewer = ctk.CTkFrame(content, fg_color="transparent")
        viewer.grid(row=0, column=1, sticky="nsew")
        viewer.columnconfigure((0, 1), weight=1)
        viewer.rowconfigure(1, weight=1)

        ctk.CTkLabel(viewer, text="Chưa đánh dấu", text_color="#b8b8b8").grid(row=0, column=0, pady=(0, 3))
        ctk.CTkLabel(viewer, text="Đã đánh dấu", text_color="#b8b8b8").grid(row=0, column=1, pady=(0, 3))

        self.lbl_raw_img = ctk.CTkLabel(viewer, text="")
        self.lbl_raw_img.grid(row=1, column=0, sticky="nsew", padx=(0, 5), pady=4)

        self.lbl_marked_img = ctk.CTkLabel(viewer, text="")
        self.lbl_marked_img.grid(row=1, column=1, sticky="nsew", padx=(5, 0), pady=4)

        slider_box = ctk.CTkFrame(self, fg_color="transparent")
        slider_box.grid(row=2, column=0, sticky="ew", pady=(4, 10))

        self.lbl_slice = ctk.CTkLabel(slider_box, text="Lát cắt: 0")
        self.lbl_slice.pack(side="left", padx=6)

        self.slider = ctk.CTkSlider(slider_box, from_=0, to=1, command=self.on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=6)
        self.slider.configure(state="disabled")

        self.bind("<Configure>", self._on_resize)

    def set_source_images(self, images):
        self.raw_images = images or []
        self.analysis_results = {}
        self._sync_slider()

    def set_analysis_results(self, analysis_results):
        self.analysis_results = analysis_results or {}
        self.display_pair()

    def set_clusters(self, clusters):
        self.clusters = clusters or []
        self.result_tree.clear()
        n_id = 1
        for c in self.clusters:
            z_val = f"{c['z_start']}-{c['z_end']}" if c['z_start'] != c['z_end'] else str(c['z_start'])
            fpr_percent = f"{c.get('fpr_score', 1.0) * 100:.1f}% AI3D"
            self.result_tree.add_item((n_id, c['voxel'], z_val, c['center_y'], c['center_x'], fpr_percent, "Red"))
            n_id += 1

    def set_fill_enabled(self, enabled):
        self.fill_enabled = bool(enabled)
        self.display_pair()

    def _sync_slider(self):
        total = len(self.raw_images)
        if total <= 0:
            self.slider.configure(state="disabled")
            self.lbl_status.configure(text="Chưa có ảnh từ tab Phân tích.", text_color="orange")
            return

        self.current_idx = 0
        steps = total - 1 if total > 1 else 1
        self.slider.configure(state="normal", from_=0, to=total - 1, number_of_steps=steps)
        self.slider.set(0)
        self.lbl_status.configure(text=f"Đang so sánh {total} lát cắt từ AI pipeline.", text_color="white")
        self.display_pair()

    def on_slider(self, value):
        self.current_idx = int(value)
        self.display_pair()

    def go_to_slice(self, z_idx):
        if not self.raw_images:
            return
        max_idx = len(self.raw_images) - 1
        self.current_idx = max(0, min(int(z_idx), max_idx))
        self.slider.set(self.current_idx)
        self.display_pair()
        # Trả focus về frame để phím trái/phải tiếp tục hoạt động sau khi click list
        self.focus_set()

    def step_slice(self, delta):
        if self.slider.cget("state") != "normal":
            return
        max_idx = len(self.raw_images) - 1
        next_idx = max(0, min(self.current_idx + int(delta), max_idx))
        if next_idx != self.current_idx:
            self.current_idx = next_idx
            self.slider.set(next_idx)
            self.display_pair()

    def _on_resize(self, event=None):
        if self.raw_images:
            self.display_pair()

    def _calc_size(self):
        left_w = self.lbl_raw_img.winfo_width()
        right_w = self.lbl_marked_img.winfo_width()
        h = min(self.lbl_raw_img.winfo_height(), self.lbl_marked_img.winfo_height())
        if left_w < 50 or right_w < 50 or h < 50:
            return 380
        return max(220, min(left_w, right_w, h) - 40)

    def display_pair(self):
        if not self.raw_images:
            return
        max_idx = len(self.raw_images) - 1
        idx = max(0, min(self.current_idx, max_idx))
        size = self._calc_size()

        raw = self.raw_images[idx]
        marked = self._build_marked_image(raw, idx)

        self.lbl_raw_img.configure(
            image=ctk.CTkImage(light_image=raw, dark_image=raw, size=(size, size)),
            text="",
        )
        self.lbl_marked_img.configure(
            image=ctk.CTkImage(light_image=marked, dark_image=marked, size=(size, size)),
            text="",
        )
        self.lbl_slice.configure(text=f"Lát cắt: {idx}")

    def _build_marked_image(self, raw_image, slice_idx):
        marked = raw_image.copy()
        res = self.analysis_results.get(slice_idx)
        if not res:
            return marked

        np_img = np.array(marked.convert("RGB"))
        for nodule in res.get("nodules", []):
            x1 = nodule.get("x1", 0)
            y1 = nodule.get("y1", 0)
            x2 = nodule.get("x2", 0)
            y2 = nodule.get("y2", 0)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = np_img[y1:y2, x1:x2]
            mask = nodule.get("fine_mask")
            if mask is None:
                cv2.rectangle(np_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                continue

            if self.fill_enabled:
                color = np.array([255, 0, 0], dtype=np.uint8)
                for c in range(3):
                    roi[:, :, c] = np.where(
                        mask,
                        (0.5 * color[c] + 0.5 * roi[:, :, c]).astype(np.uint8),
                        roi[:, :, c],
                    )
            else:
                mask_uint8 = mask.astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(roi, contours, -1, (255, 0, 0), 1)
            np_img[y1:y2, x1:x2] = roi
        return Image.fromarray(np_img)