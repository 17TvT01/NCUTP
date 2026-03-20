import customtkinter as ctk
from PIL import Image
import numpy as np
import cv2

class ImageViewer(ctk.CTkFrame):
    def __init__(self, master, settings_panel, **kwargs):
        super().__init__(master, fg_color="#2b2b2b", border_width=1, border_color="#555555", **kwargs)
        self.ct_images = []
        self.analysis_results = {}
        self.current_slice_index = 0
        self.settings = settings_panel
        
        self.setup_ui()

    def setup_ui(self):
        sld_frame = ctk.CTkFrame(self, fg_color="transparent")
        sld_frame.pack(side="bottom", fill="x", padx=10, pady=(0, 10))
        self.lbl_slice = ctk.CTkLabel(sld_frame, text="Lát cắt: 0", width=80)
        self.lbl_slice.pack(side="left", padx=5)
        
        self.slider = ctk.CTkSlider(sld_frame, from_=0, to=1, command=self.on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=5)
        self.slider.set(0)
        self.slider.configure(state="disabled")

        self.img_lbl = ctk.CTkLabel(self, text="")
        self.img_lbl.pack(fill="both", expand=True, padx=5, pady=5)
        self.bind("<Configure>", self._on_img_frame_resize)

    def set_images(self, imgs):
        self.ct_images = imgs
        if imgs:
            self.slider.configure(state="normal", from_=0, to=len(imgs)-1, number_of_steps=len(imgs)-1)
            self.slider.set(0)
            self.current_slice_index = 0
            self.display_slice()
            
    def set_analysis_results(self, results):
        self.analysis_results = results
        self.display_slice()
        
    def update_analysis_result(self, slice_idx, res):
        """Update just one slice result as it comes in during streaming analysis"""
        self.analysis_results[slice_idx] = res

    def get_images(self):
        return self.ct_images

    def on_slider(self, val):
        if self.ct_images: 
            self.current_slice_index = int(val)
            self.display_slice()

    def go_to_slice(self, z_idx):
        if 0 <= z_idx < len(self.ct_images): 
            self.slider.set(z_idx)
            self.on_slider(z_idx)

    def disable_slider(self):
        self.slider.configure(state="disabled")

    def enable_slider(self):
        if self.ct_images:
            self.slider.configure(state="normal")
            
    def _get_display_size(self):
        w = self.img_lbl.winfo_width()
        h = self.img_lbl.winfo_height()
        if w < 50 or h < 50:
            return 350
        size = max(200, min(w, h) - 60)
        return min(size, 550)

    def _on_img_frame_resize(self, event=None):
        if self.ct_images:
            self.display_slice()

    def display_slice(self):
        if not self.ct_images: return
        img_orig = self.ct_images[self.current_slice_index]
        if self.current_slice_index in self.analysis_results:
            res = self.analysis_results[self.current_slice_index]
            np_img = np.array(img_orig.convert("RGB"))
            
            fill_enabled = self.settings.get_fill_color_enabled()
            
            for n in res.get("nodules", []):
                x1, y1, x2, y2 = n["x1"], n["y1"], n["x2"], n["y2"]
                if n.get("fine_mask") is not None:
                    roi = np_img[y1:y2, x1:x2]
                    if fill_enabled:
                        for c in range(3):
                            roi[:,:,c] = np.where(n["fine_mask"], (0.5 * np.array([255,0,0])[c] + 0.5 * roi[:,:,c]).astype(np.uint8), roi[:,:,c])
                    else:
                        mask_uint8 = n["fine_mask"].astype(np.uint8)
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(roi, contours, -1, (255, 0, 0), 1)
                    np_img[y1:y2, x1:x2] = roi
            img_final = Image.fromarray(np_img)
        else: img_final = img_orig.copy()
        
        display_size = self._get_display_size()
        self.img_lbl.configure(image=ctk.CTkImage(light_image=img_final, dark_image=img_final, size=(display_size, display_size)), text="")
        self.lbl_slice.configure(text=f"Lát cắt: {self.current_slice_index}")
