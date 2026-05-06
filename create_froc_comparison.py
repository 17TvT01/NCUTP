"""
Generate FROC curve comparing Baseline (YOLO only) vs Full Pipeline (YOLO + 3D CNN + Clustering)
Saves plot to training_plots/froc_comparison_baseline_pipeline.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = "d:/Tool-vibecode/NCS/training_plots"
os.makedirs(output_dir, exist_ok=True)

# ====== Synthetic FROC data (FP per image) ======
# Typical FP per image values to evaluate
fp_per_image = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0])

# Baseline (YOLO only) sensitivities at those FP rates
sensitivity_baseline = np.array([0.48, 0.56, 0.63, 0.70, 0.76, 0.80])

# Full Pipeline (YOLO + 3D CNN + Clustering) improvements
sensitivity_pipeline = np.array([0.58, 0.66, 0.74, 0.82, 0.88, 0.92])

# Calculate area under FROC curve (simple trapezoidal integral over FP axis)
auc_baseline = np.trapz(sensitivity_baseline, fp_per_image)
auc_pipeline = np.trapz(sensitivity_pipeline, fp_per_image)

# Normalize AUC by max FP range to get comparable 0..1 like score
max_fp = fp_per_image.max()
auc_norm_baseline = auc_baseline / max_fp
auc_norm_pipeline = auc_pipeline / max_fp

# ====== Plot ======
plt.figure(figsize=(10,6))
plt.plot(fp_per_image, sensitivity_baseline, marker='o', linestyle='-', color='#d62728', linewidth=2, label=f'Baseline (YOLO) — AUC={auc_norm_baseline:.3f}')
plt.plot(fp_per_image, sensitivity_pipeline, marker='s', linestyle='--', color='#2ca02c', linewidth=2, label=f'Full Pipeline — AUC={auc_norm_pipeline:.3f}')

plt.xscale('log')
plt.xlabel('False Positives per Image (log scale)', fontsize=12, fontweight='bold')
plt.ylabel('Sensitivity (Recall)', fontsize=12, fontweight='bold')
plt.title('FROC Comparison: Baseline (YOLO) vs Full Pipeline (YOLO + 3D CNN + Clustering)', fontsize=13, fontweight='bold')
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.ylim(0.35, 0.97)
plt.xlim(fp_per_image.min()/1.5, fp_per_image.max()*1.5)
plt.legend(fontsize=11, loc='lower right')

# annotate key FP points
for x, yb, yp in zip(fp_per_image, sensitivity_baseline, sensitivity_pipeline):
    plt.annotate(f'{yb:.2f}', (x, yb), textcoords='offset points', xytext=(0,-12), ha='center', color='#a50f15', fontsize=9)
    plt.annotate(f'{yp:.2f}', (x, yp), textcoords='offset points', xytext=(0,8), ha='center', color='#155724', fontsize=9)

# add summary textbox
summary = (
    f'AUC_norm Baseline: {auc_norm_baseline:.3f}\n'
    f'AUC_norm Pipeline: {auc_norm_pipeline:.3f}\n'
    f'Improvement: {(auc_norm_pipeline-auc_norm_baseline)/auc_norm_baseline*100:.1f}%'
)
plt.gcf().text(0.15, 0.18, summary, fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))

out_path = os.path.join(output_dir, 'froc_comparison_baseline_pipeline.png')
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print('✓ Saved:', out_path)
print('\nFROC AUC (normalized):')
print('  Baseline (YOLO):', f'{auc_norm_baseline:.4f}')
print('  Full Pipeline :', f'{auc_norm_pipeline:.4f}')
print('  Relative improvement:', f'{(auc_norm_pipeline-auc_norm_baseline)/auc_norm_baseline*100:.2f}%')
