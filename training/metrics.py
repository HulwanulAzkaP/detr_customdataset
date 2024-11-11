import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_metrics(pred_boxes, pred_labels, target_boxes, target_labels):
    """
    Calculate Precision, Recall, F1-Score for object detection.
    """
    # Pastikan pred_labels dan target_labels tidak kosong sebelum menggabungkan
    if len(pred_labels) == 0 or len(target_labels) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }

    # Flatten the predictions and targets
    pred_labels = torch.cat([label for label in pred_labels if label.numel() > 0]).cpu().numpy()
    target_labels = torch.cat([label for label in target_labels if label.numel() > 0]).cpu().numpy()

    # Pastikan jumlah prediksi dan target sama
    min_length = min(len(pred_labels), len(target_labels))
    pred_labels = pred_labels[:min_length]
    target_labels = target_labels[:min_length]

    # Hitung Precision, Recall, dan F1-Score menggunakan sklearn
    precision = precision_score(target_labels, pred_labels, average='weighted', zero_division=1)
    recall = recall_score(target_labels, pred_labels, average='weighted', zero_division=1)
    f1 = f1_score(target_labels, pred_labels, average='weighted', zero_division=1)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
