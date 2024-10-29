from ultralytics import YOLO
from comet_ml import Experiment
import torch
import numpy as np

experiment = Experiment(
    api_key="I8wo6R02iil0YFRWN7ksASxsR",
    project_name="enhancing-home-security",
    workspace="chiragagg5k",
)

model_name = "yolo11l"
model = YOLO(f"models/{model_name}.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
model = model.to(device)

iou_thresholds = np.arange(0.5, 0.95, 0.05)

val_data = "dangerous-data.yaml"

for iou_threshold in iou_thresholds:
    print(f"\n--- Validating at IoU threshold: {iou_threshold:.2f} ---")
    
    results = model.val(
        data=val_data,
        imgsz=640,
        batch=16,
        iou=iou_threshold,
        verbose=True
    )
    
    metrics = results.results_dict
    precision = metrics["metrics/precision(B)"]
    recall = metrics["metrics/recall(B)"]
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")
    
    experiment.log_metrics({
        f"precision_IoU_{iou_threshold:.2f}": precision,
        f"recall_IoU_{iou_threshold:.2f}": recall,
        f"F1-score_IoU_{iou_threshold:.2f}": f1_score
    })

    experiment.log_image(f"Confusion_Matrix_IoU_{iou_threshold:.2f}", results.confusion_matrix.plot(normalize=True))
    experiment.log_image(f"PR_Curve_IoU_{iou_threshold:.2f}", results.plot_pr_curve())

experiment.end()

print("\nValidation completed for all IoU thresholds.")