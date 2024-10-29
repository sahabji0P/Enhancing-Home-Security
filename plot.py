import matplotlib.pyplot as plt
import numpy as np

# Updated data from the table
thresholds = ['base', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
map_50 = [0.868, 0.884, 0.891, 0.896, 0.900, 0.902, 0.901, 0.897, 0.889, 0.856]
map_50_95 = [0.710, 0.720, 0.724, 0.727, 0.729, 0.730, 0.730, 0.727, 0.722, 0.701]
precision = [0.934, 0.931, 0.930, 0.929, 0.925, 0.929, 0.918, 0.933, 0.932, 0.871]
recall = [0.789, 0.815, 0.822, 0.829, 0.832, 0.830, 0.838, 0.822, 0.814, 0.799]
f1_score = [0.856, 0.869, 0.873, 0.876, 0.876, 0.877, 0.876, 0.874, 0.869, 0.833]

# Create figure and axis with a larger size
plt.figure(figsize=(14, 7))  # Increased width to accommodate more data points

# Create x-axis positions
x = np.arange(len(thresholds))

# Plot lines with markers
plt.plot(x, map_50, 'o-', label='mAP@0.5', linewidth=2, markersize=8)
plt.plot(x, map_50_95, 's-', label='mAP@0.5:0.95', linewidth=2, markersize=8)
plt.plot(x, precision, '^-', label='Precision', linewidth=2, markersize=8)
plt.plot(x, recall, 'D-', label='Recall', linewidth=2, markersize=8)
plt.plot(x, f1_score, 'v-', label='F1 Score', linewidth=2, markersize=8)

# Highlight IoU 0.5 with vertical line and points
iou_05_index = thresholds.index('0.5')
plt.axvline(x=iou_05_index, color='lightgray', linestyle='--', alpha=0.5, linewidth=2)

# Add larger markers for IoU 0.5 points
plt.plot(iou_05_index, map_50[iou_05_index], 'o', color='red', markersize=12, label='IoU 0.5 points')
plt.plot(iou_05_index, map_50_95[iou_05_index], 's', color='red', markersize=12)
plt.plot(iou_05_index, precision[iou_05_index], '^', color='red', markersize=12)
plt.plot(iou_05_index, recall[iou_05_index], 'D', color='red', markersize=12)
plt.plot(iou_05_index, f1_score[iou_05_index], 'v', color='red', markersize=12)

# Customize the plot
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('NMS IoU Threshold', fontsize=12, fontweight='bold')
plt.ylabel('Score', fontsize=12, fontweight='bold')
# plt.title('Impact of NMS IoU Threshold on YOLO11m Model Metrics\n(IoU 0.5 Highlighted)', fontsize=14, fontweight='bold', pad=20)

# Set x-axis ticks with rotation for better readability
plt.xticks(x, thresholds, rotation=45)

# Add legend
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Add minor gridlines
plt.grid(True, which='minor', linestyle=':', alpha=0.4)

# Set y-axis limits with some padding
plt.ylim(0.65, 1.0)  # Adjusted to accommodate lower values

# Add value annotations with special formatting for IoU 0.5
for metric in [map_50, map_50_95, precision, recall, f1_score]:
    for i, value in enumerate(metric):
        if i == iou_05_index:
            # Highlighted annotation for IoU 0.5
            plt.annotate(f'{value:.3f}', 
                        (x[i], value),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=9,
                        fontweight='bold',
                        color='red')
        else:
            # Regular annotations
            plt.annotate(f'{value:.3f}', 
                        (x[i], value),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=8)

# Adjust layout to prevent label cutoff
plt.tight_layout()

plt.show()