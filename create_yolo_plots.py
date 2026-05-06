"""
Script tạo biểu đồ mAP@0.5, Loss Boxes cho YOLO v8 vs v11 training
Từ dữ liệu thực tế trong results.csv
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ==================== Load Data từ results.csv ====================
results_file = "d:/Tool-vibecode/NCS/runs_compare/train_yolov11/results.csv"
df = pd.read_csv(results_file)

# Tạo folder output
output_dir = "d:/Tool-vibecode/NCS/training_plots"
os.makedirs(output_dir, exist_ok=True)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# ==================== 1. VẼ mAP@0.5 (Box Detection) ====================
fig, ax = plt.subplots(figsize=(10, 6))

epochs = df['epoch'].values
map50 = df['metrics/mAP50(B)'].values * 100  # Convert to percentage

ax.plot(epochs, map50, 'b-', linewidth=2.5, label='mAP@0.5 (YOLOv11n)', marker='o', markersize=4, alpha=0.8)
ax.fill_between(epochs, map50, alpha=0.2, color='blue')

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('mAP@0.5 (%)', fontsize=12, fontweight='bold')
ax.set_title('YOLO v11: mAP@0.5 Improvement (Nodule Detection)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')

# Annotation cho best mAP
best_map_idx = np.argmax(map50)
best_map = map50[best_map_idx]
ax.annotate(f'Best mAP: {best_map:.2f}%\n(Epoch {int(epochs[best_map_idx])})',
            xy=(epochs[best_map_idx], best_map),
            xytext=(epochs[best_map_idx]-5, best_map-5),
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.7', facecolor='yellow', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'yolo_map50_curve.png'), dpi=300, bbox_inches='tight')
print("\n✓ Saved: yolo_map50_curve.png")
plt.close()

# ==================== 2. VẼ Loss Boxes (Training vs Validation) ====================
fig, ax = plt.subplots(figsize=(11, 6))

train_box_loss = df['train/box_loss'].values
val_box_loss = df['val/box_loss'].values

ax.plot(epochs, train_box_loss, 'g-', linewidth=2.5, label='Training Box Loss', marker='o', markersize=4, alpha=0.8)
ax.plot(epochs, val_box_loss, 'r-', linewidth=2.5, label='Validation Box Loss', marker='s', markersize=4, alpha=0.8)

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Box Loss', fontsize=12, fontweight='bold')
ax.set_title('YOLO v11: Box Loss Curve (Training vs Validation)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')

# Annotation cho best val loss
best_val_loss_idx = np.argmin(val_box_loss)
best_val_loss = val_box_loss[best_val_loss_idx]
ax.annotate(f'Best Val Loss: {best_val_loss:.4f}\n(Epoch {int(epochs[best_val_loss_idx])})',
            xy=(epochs[best_val_loss_idx], best_val_loss),
            xytext=(epochs[best_val_loss_idx]+3, best_val_loss+0.15),
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.7', facecolor='lightcoral', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'yolo_box_loss_curve.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: yolo_box_loss_curve.png")
plt.close()

# ==================== 3. VẼ Precision, Recall, mAP50-95 ====================
fig, ax = plt.subplots(figsize=(11, 6))

precision = df['metrics/precision(B)'].values * 100
recall = df['metrics/recall(B)'].values * 100
map50_95 = df['metrics/mAP50-95(B)'].values * 100

ax.plot(epochs, precision, 'g-', linewidth=2, label='Precision', marker='o', markersize=4, alpha=0.8)
ax.plot(epochs, recall, 'b-', linewidth=2, label='Recall', marker='s', markersize=4, alpha=0.8)
ax.plot(epochs, map50_95, 'r-', linewidth=2, label='mAP@0.5-0.95', marker='^', markersize=4, alpha=0.8)

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('YOLO v11: Precision, Recall, mAP@0.5-0.95', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0, 110)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'yolo_precision_recall_map.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: yolo_precision_recall_map.png")
plt.close()

# ==================== 4. VẼ Tất cả Loss (Box, Class, DFL) ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Training losses
train_cls_loss = df['train/cls_loss'].values
train_dfl_loss = df['train/dfl_loss'].values

ax1.plot(epochs, train_box_loss, 'r-', linewidth=2, label='Box Loss', marker='o', markersize=3, alpha=0.8)
ax1.plot(epochs, train_cls_loss, 'g-', linewidth=2, label='Class Loss', marker='s', markersize=3, alpha=0.8)
ax1.plot(epochs, train_dfl_loss, 'b-', linewidth=2, label='DFL Loss', marker='^', markersize=3, alpha=0.8)

ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax1.set_title('Training Losses (Box, Class, DFL)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3, linestyle='--')

# Validation losses
val_cls_loss = df['val/cls_loss'].values
val_dfl_loss = df['val/dfl_loss'].values

ax2.plot(epochs, val_box_loss, 'r-', linewidth=2, label='Box Loss', marker='o', markersize=3, alpha=0.8)
ax2.plot(epochs, val_cls_loss, 'g-', linewidth=2, label='Class Loss', marker='s', markersize=3, alpha=0.8)
ax2.plot(epochs, val_dfl_loss, 'b-', linewidth=2, label='DFL Loss', marker='^', markersize=3, alpha=0.8)

ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax2.set_title('Validation Losses (Box, Class, DFL)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'yolo_all_losses_training.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: yolo_all_losses_training.png")
plt.close()

# ==================== 5. VẼ So sánh giai đoạn ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Giai đoạn (Early, Mid, Late)
stages = ['Early\n(Epoch 5)', 'Mid\n(Epoch 25)', 'Late\n(Epoch 50)']
early_idx, mid_idx, late_idx = 4, 24, min(49, len(epochs)-1)

# --- Subplot 1: Box Loss Progression ---
train_box_vals = [train_box_loss[early_idx], train_box_loss[mid_idx], train_box_loss[late_idx]]
val_box_vals = [val_box_loss[early_idx], val_box_loss[mid_idx], val_box_loss[late_idx]]

x = np.arange(len(stages))
width = 0.35

bars1 = axes[0, 0].bar(x - width/2, train_box_vals, width, label='Training', color='#3498db', alpha=0.8)
bars2 = axes[0, 0].bar(x + width/2, val_box_vals, width, label='Validation', color='#e74c3c', alpha=0.8)

axes[0, 0].set_ylabel('Box Loss', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Box Loss Progression', fontsize=12, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(stages)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# --- Subplot 2: mAP Progression ---
map50_vals = [map50[early_idx], map50[mid_idx], map50[late_idx]]

bars = axes[0, 1].bar(x, map50_vals, width*2, color=['#e67e22', '#f39c12', '#2ecc71'], alpha=0.8)

axes[0, 1].set_ylabel('mAP@0.5 (%)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('mAP@0.5 Improvement', fontsize=12, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(stages)
axes[0, 1].set_ylim(0, max(map50_vals)*1.2)
axes[0, 1].grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# --- Subplot 3: Precision & Recall ---
precision_vals = [precision[early_idx], precision[mid_idx], precision[late_idx]]
recall_vals = [recall[early_idx], recall[mid_idx], recall[late_idx]]

bars1 = axes[1, 0].bar(x - width/2, precision_vals, width, label='Precision', color='#2ecc71', alpha=0.8)
bars2 = axes[1, 0].bar(x + width/2, recall_vals, width, label='Recall', color='#3498db', alpha=0.8)

axes[1, 0].set_ylabel('Score (%)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Precision & Recall Progression', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(stages)
axes[1, 0].set_ylim(0, 110)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# --- Subplot 4: Training Efficiency ---
time_vals = [df['time'].iloc[early_idx], df['time'].iloc[mid_idx], df['time'].iloc[late_idx]]
time_cumulative = [time_vals[0], time_vals[0]+time_vals[1], time_vals[0]+time_vals[1]+time_vals[2]]

bars = axes[1, 1].bar(stages, time_cumulative, color=['#9b59b6', '#8e44ad', '#6c3483'], alpha=0.8)

axes[1, 1].set_ylabel('Cumulative Time (seconds)', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Training Time (Cumulative)', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'yolo_training_stages_detailed.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: yolo_training_stages_detailed.png")
plt.close()

# ==================== 6. IN RA THỐNG KÊ ====================
print("\n" + "="*70)
print("YOLO V11 TRAINING SUMMARY")
print("="*70)
print(f"Total Epochs: {len(epochs)}")
print(f"Total Training Time: {df['time'].iloc[-1]:.1f} seconds ({df['time'].iloc[-1]/60:.1f} minutes)")
print()
print("BOX LOSS:")
print(f"  Initial Training: {train_box_loss[0]:.4f}")
print(f"  Final Training:   {train_box_loss[-1]:.4f} (↓ {(1-train_box_loss[-1]/train_box_loss[0])*100:.1f}%)")
print(f"  Best Validation:  {np.min(val_box_loss):.4f} (Epoch {np.argmin(val_box_loss)+1})")
print()
print("mAP@0.5:")
print(f"  Best mAP@0.5:     {best_map:.2f}% (Epoch {int(epochs[best_map_idx])})")
print(f"  Final mAP@0.5:    {map50[-1]:.2f}%")
print()
print("PRECISION & RECALL:")
print(f"  Final Precision:  {precision[-1]:.2f}%")
print(f"  Final Recall:     {recall[-1]:.2f}%")
print()
print("mAP@0.5-0.95:")
print(f"  Final mAP@0.5-95: {map50_95[-1]:.2f}%")
print("="*70)
print(f"\nAll YOLO plots saved to: {output_dir}")
print("✓ Tất cả ảnh đã sẵn sàng cho bài viết!")
