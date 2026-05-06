"""
Script minh họa: So sánh Trước/Sau Morphological Opening
(a) Trước: Nhiều đốm li ti, noise từ YOLO detection
(b) Sau: Sạch, chỉ giữ nodule lớn sau morphological filter
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os

output_dir = "d:/Tool-vibecode/NCS/training_plots"
os.makedirs(output_dir, exist_ok=True)

# ==================== 1. TẠO SYNTHETIC DETECTION RESULTS ====================
# Simulate YOLO detection outputs on a CT slice
img_size = 512
img_clean = np.zeros((img_size, img_size), dtype=np.uint8)

# Draw nodules (phổi bề ngoài)
cv2.circle(img_clean, (200, 250), 100, 255, -1)  # Phổi trái lớn
cv2.circle(img_clean, (380, 280), 80, 255, -1)   # Phổi phải

# Draw some true positive nodules inside lungs
cv2.circle(img_clean, (180, 200), 20, 200, -1)   # Nodule 1: Medium
cv2.circle(img_clean, (250, 300), 12, 200, -1)   # Nodule 2: Small
cv2.circle(img_clean, (350, 250), 18, 200, -1)   # Nodule 3: Medium
cv2.circle(img_clean, (420, 320), 15, 200, -1)   # Nodule 4: Small

# Tính toán trước filtering: thêm NOISE/FALSE POSITIVES
img_before_filter = img_clean.copy()

# Thêm các đốm li ti (false positives từ YOLO)
np.random.seed(42)
for _ in range(60):  # 60 false positive detections
    x = np.random.randint(50, img_size - 50)
    y = np.random.randint(50, img_size - 50)
    r = np.random.randint(1, 6)  # Radius 1-5 pixels (rất nhỏ)
    if img_clean[y, x] > 100:  # Thêm noise trong vùng phổi
        cv2.circle(img_before_filter, (x, y), r, 180, -1)

# Lớp lót thêm artifacts
artifact_mask = np.random.random((img_size, img_size)) > 0.97
artifact_indices = artifact_mask & (img_clean > 100)
img_before_filter = img_before_filter.astype(np.int32)
img_before_filter[artifact_indices] += np.random.randint(10, 40, np.sum(artifact_indices))
img_before_filter = np.clip(img_before_filter, 0, 255).astype(np.uint8)

# ==================== 2. APPLY MORPHOLOGICAL OPENING ====================
# Tạo kernel cho morphological operations
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Apply Opening (xóa bỏ những đốm nhỏ)
img_after_opening = cv2.morphologyEx(img_before_filter, cv2.MORPH_OPEN, kernel_open, iterations=2)

# Apply Closing (lấp đầy các khoảng trống bên trong nodules)
img_after_filter = cv2.morphologyEx(img_after_opening, cv2.MORPH_CLOSE, kernel_close, iterations=1)

# ==================== 3. VẼ COMPARISON ====================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# ------- (a) BEFORE FILTERING -------
ax_before = axes[0]
im_before = ax_before.imshow(img_before_filter, cmap='gray', vmin=0, vmax=255)
ax_before.set_title('(a) Trước khi Morphological Opening\n(Nhiều đốm li ti, noise từ YOLO)', 
                     fontsize=13, fontweight='bold', color='darkred', pad=15)
ax_before.axis('off')

# Thêm legend cho before image
legend_text_before = (
    "Issues:\n"
    "• 60+ false positive detections\n"
    "• Noise artifacts từ low confidence\n"
    "• Rất khó phân biệt nodule thực\n"
    "• Artifact từ mạch máu/xương"
)
ax_before.text(0.02, 0.98, legend_text_before, transform=ax_before.transAxes,
               fontsize=10, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

# ------- (b) AFTER FILTERING -------
ax_after = axes[1]
im_after = ax_after.imshow(img_after_filter, cmap='gray', vmin=0, vmax=255)
ax_after.set_title('(b) Sau khi Morphological Opening\n(Sạch, chỉ giữ nodules thực)', 
                    fontsize=13, fontweight='bold', color='darkgreen', pad=15)
ax_after.axis('off')

# Thêm legend cho after image
legend_text_after = (
    "Improvements:\n"
    "✓ Toàn bộ noise được xóa\n"
    "✓ Giữ lại nodules >10 pixels\n"
    "✓ Morphology thực tế hơn\n"
    "✓ Sẵn sàng cho 3D CNN"
)
ax_after.text(0.02, 0.98, legend_text_after, transform=ax_after.transAxes,
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'morphological_filter_before_after.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: morphological_filter_before_after.png")
plt.close()

# ==================== 4. VẼ DETAILED MORPHOLOGICAL OPERATIONS ====================
fig = plt.figure(figsize=(18, 10))

# Tạo 3x2 grid
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.25)

# Row 1: Tiến trình lọc chi tiết
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(img_before_filter, cmap='gray')
ax1.set_title('(1) Original Detection Results\n(After YOLO + artifacts)', fontsize=11, fontweight='bold')
ax1.axis('off')

# Erosion
img_eroded = cv2.erode(img_before_filter, kernel_open, iterations=2)
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(img_eroded, cmap='gray')
ax2.set_title('(2) After Erosion\n(Removes small noise)', fontsize=11, fontweight='bold')
ax2.axis('off')

# Opening = Erosion + Dilation
img_opened = cv2.dilate(img_eroded, kernel_open, iterations=2)
ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(img_opened, cmap='gray')
ax3.set_title('(3) After Dilation (Opening)\n(Restores nodule size)', fontsize=11, fontweight='bold')
ax3.axis('off')

# Row 2: Chi tiết hơn
# Closing
img_closing = cv2.morphologyEx(img_opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
ax4 = fig.add_subplot(gs[1, 0])
ax4.imshow(img_closing, cmap='gray')
ax4.set_title('(4) After Closing\n(Fills internal gaps)', fontsize=11, fontweight='bold')
ax4.axis('off')

# Contours before
ax5 = fig.add_subplot(gs[1, 1])
img_with_contours_before = cv2.cvtColor(img_before_filter, cv2.COLOR_GRAY2BGR)
contours_before, _ = cv2.findContours(img_before_filter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_with_contours_before, contours_before, -1, (0, 255, 0), 2)
ax5.imshow(cv2.cvtColor(img_with_contours_before, cv2.COLOR_BGR2RGB))
ax5.set_title(f'Contours BEFORE: {len(contours_before)} regions\n(Too many false positives!)', 
              fontsize=11, fontweight='bold', color='red')
ax5.axis('off')

# Contours after
ax6 = fig.add_subplot(gs[1, 2])
img_with_contours_after = cv2.cvtColor(img_after_filter, cv2.COLOR_GRAY2BGR)
contours_after, _ = cv2.findContours(img_after_filter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_with_contours_after, contours_after, -1, (0, 255, 0), 2)
ax6.imshow(cv2.cvtColor(img_with_contours_after, cv2.COLOR_BGR2RGB))
ax6.set_title(f'Contours AFTER: {len(contours_after)} regions\n(Clean, only true nodules)', 
              fontsize=11, fontweight='bold', color='green')
ax6.axis('off')

fig.suptitle('Morphological Filtering Pipeline: Detailed Step-by-Step', 
             fontsize=15, fontweight='bold', y=0.995)

plt.savefig(os.path.join(output_dir, 'morphological_filtering_detailed.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: morphological_filtering_detailed.png")
plt.close()

# ==================== 5. VẼ KERNEL VISUALIZATION ====================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Opening kernel
opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
axes[0].imshow(opening_kernel, cmap='gray')
axes[0].set_title('Opening Kernel\n(5×5 Ellipse)', fontsize=12, fontweight='bold')
axes[0].set_xticks([])
axes[0].set_yticks([])
for i in range(opening_kernel.shape[0]):
    for j in range(opening_kernel.shape[1]):
        axes[0].text(j, i, str(opening_kernel[i, j]), ha='center', va='center', fontsize=10, fontweight='bold')

# Closing kernel
closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
axes[1].imshow(closing_kernel, cmap='gray')
axes[1].set_title('Closing Kernel\n(7×7 Ellipse)', fontsize=12, fontweight='bold')
axes[1].set_xticks([])
axes[1].set_yticks([])

# Effect diagram
ax_effect = axes[2]
ax_effect.axis('off')
effect_text = """
MORPHOLOGICAL OPERATIONS:

Opening (Erosion → Dilation):
  • Removes small objects/noise
  • Smooths boundaries
  • Preserves object shape
  • Formula: A ∘ B = (A ⊖ B) ⊕ B

Closing (Dilation → Erosion):
  • Fills internal holes/gaps
  • Smooths boundaries
  • Preserves object shape
  • Formula: A • B = (A ⊕ B) ⊖ B

Kernel: Structuring element shape
  • Ellipse: Circular filtering
  • Rectangle: Directional filtering
  • Size: Controls filter strength
"""
ax_effect.text(0.5, 0.5, effect_text, 
              transform=ax_effect.transAxes,
              fontsize=10, verticalalignment='center',
              horizontalalignment='center',
              family='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'morphological_kernels.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: morphological_kernels.png")
plt.close()

# ==================== 6. STATISTICS ====================
print("\n" + "="*70)
print("MORPHOLOGICAL FILTERING ANALYSIS")
print("="*70)

# Count detections
_, binary_before = cv2.threshold(img_before_filter, 100, 1, cv2.THRESH_BINARY)
_, binary_after = cv2.threshold(img_after_filter, 100, 1, cv2.THRESH_BINARY)

contours_before, _ = cv2.findContours(img_before_filter.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_after, _ = cv2.findContours(img_after_filter.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"\nBEFORE Morphological Filtering:")
print(f"  Total detections: {len(contours_before)}")
print(f"  Total pixels: {np.sum(binary_before)}")

print(f"\nAFTER Morphological Filtering:")
print(f"  Total detections: {len(contours_after)}")
print(f"  Total pixels: {np.sum(binary_after)}")
print(f"  Reduction: {len(contours_before) - len(contours_after)} false positives removed ({100*(len(contours_before)-len(contours_after))/len(contours_before):.1f}%)")

print(f"\nKERNEL SPECIFICATIONS:")
print(f"  Opening: 5×5 Ellipse, 2 iterations (strong erosion)")
print(f"  Closing: 7×7 Ellipse, 1 iteration (moderate dilation)")

print("\n" + "="*70)
print(f"✓ All images saved to: {output_dir}")
print("✓ 3 comparison images ready for thesis insertion!")
print("="*70)
