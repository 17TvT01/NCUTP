"""
Create two error-case illustrative images:
(a) Very faint GGN (missed) — no bounding box
(b) Vessel bifurcation misdetected as nodule — show false positive box
Saves to training_plots/error_cases_gnn_vessel.png
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

output_dir = "d:/Tool-vibecode/NCS/training_plots"
os.makedirs(output_dir, exist_ok=True)

img_size = 512

# Helper: create lung background
def make_lung_background(size=img_size):
    bg = np.zeros((size, size), dtype=np.uint8) + 20
    cv2.circle(bg, (size//2 - 70, size//2), 140, 80, -1)
    cv2.circle(bg, (size//2 + 70, size//2), 140, 80, -1)
    # lung texture noise
    noise = (np.random.randn(size, size) * 3).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return bg

# (a) Very faint GGN (missed)
bg_a = make_lung_background()
# create faint GGN as low-contrast blurred circular patch
ggn_center = (img_size//2 - 70, img_size//2 - 20)
ggn = np.zeros_like(bg_a)
cv2.circle(ggn, ggn_center, 18, 40, -1)  # low intensity
ggn = cv2.GaussianBlur(ggn, (31,31), 0)
img_a = cv2.add(bg_a, ggn)
# annotate that it's missed (no box)

# (b) Vessel bifurcation misdetected
bg_b = make_lung_background()
# draw vessel branches (bright linear structures)
v = bg_b.copy()
cv2.line(v, (img_size//2 + 20, img_size//2 - 60), (img_size//2 + 20, img_size//2 + 20), 200, 6)
cv2.line(v, (img_size//2 + 20, img_size//2 + 20), (img_size//2 + 60, img_size//2 + 60), 200, 6)
cv2.line(v, (img_size//2 + 20, img_size//2 + 20), (img_size//2 - 10, img_size//2 + 60), 200, 6)
# add bright blob at bifurcation
cv2.circle(v, (img_size//2 + 20, img_size//2 + 20), 10, 220, -1)
v = cv2.GaussianBlur(v, (5,5), 0)
img_b = v

# Simulate detection: draw a false positive bounding box around bifurcation
fp_box_pt1 = (img_size//2 + 8, img_size//2 + 8)
fp_box_pt2 = (img_size//2 + 36, img_size//2 + 36)
cv2.rectangle(img_b, fp_box_pt1, fp_box_pt2, (255), 2)  # white box will show on grayscale later

# Convert to RGB for plotting
img_a_rgb = cv2.cvtColor(img_a, cv2.COLOR_GRAY2RGB)
img_b_rgb = cv2.cvtColor(img_b, cv2.COLOR_GRAY2RGB)
# draw red box overlay for FP (in RGB)
cv2.rectangle(img_b_rgb, fp_box_pt1, fp_box_pt2, (220, 30, 30), 2)
cv2.putText(img_b_rgb, 'FP: Vessel', (fp_box_pt1[0], fp_box_pt1[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,30,30), 2)

# Compose figure
fig, axes = plt.subplots(1,2, figsize=(14,7))
axes[0].imshow(img_a_rgb, cmap='gray')
axes[0].axis('off')
axes[0].set_title("(a) Faint GGN missed by detector", fontsize=12, fontweight='bold')
axes[0].text(0.02, 0.95, "Problem: Low contrast GGN (ground-glass)\n- Small amplitude vs background\n- Model false negative", transform=axes[0].transAxes, fontsize=10, va='top', bbox=dict(facecolor='wheat', alpha=0.9))

axes[1].imshow(img_b_rgb)
axes[1].axis('off')
axes[1].set_title("(b) Vessel bifurcation misdetected as nodule", fontsize=12, fontweight='bold')
axes[1].text(0.02, 0.95, 'Problem: Vascular structure with bulbous junction\n- High local intensity and circular shape\n- Model false positive', transform=axes[1].transAxes, fontsize=10, va='top', bbox=dict(facecolor='wheat', alpha=0.9))

fig.suptitle('Error Cases: Missed GGN and Vessel False Positive', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96])
out_path = os.path.join(output_dir, 'error_cases_gnn_vessel.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print('✓ Saved:', out_path)
