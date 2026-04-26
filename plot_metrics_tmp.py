import os
import matplotlib.pyplot as plt
import numpy as np

output_dir = r"C:\Users\Admin\.gemini\antigravity\brain\9a3a3b9b-ee36-4460-9d40-cde902d5fbfa"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Data
labels = ['Pre-Optimization (Old App)', 'Post-Optimization (Ensemble+TTA+CLAHE)']
recall = [30.76, 100.0]
precision = [80.0, 7.5]
true_positives = [8, 26]
false_negatives = [18, 0]

x = np.arange(len(labels))
width = 0.35

# 1. Recall Chart
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(x, recall, width, color=['#e74c3c', '#2ecc71'])
ax.set_ylabel('Recall (%)')
ax.set_title('Lung Nodule Detection: Recall Improvement')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 110)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'recall_chart.png'), dpi=300)
plt.close()

# 2. Missed Nodules Chart (False Negatives)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(x, false_negatives, width, color=['#e67e22', '#3498db'])
ax.set_ylabel('Number of Missed Nodules')
ax.set_title('Dangerous Misses (False Negatives)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 25)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval} Nodules', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fn_chart.png'), dpi=300)
plt.close()

print("Charts generated successfully in artifact directory.")
