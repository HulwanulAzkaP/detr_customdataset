import torch
from tqdm import tqdm
from training.loss import compute_loss
from training.metrics import calculate_metrics
from config.config import Config


def train_one_epoch(model, dataloader, optimizer, device, training=True):
    """
    Train or validate the model for one epoch.
    """
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    for images, targets in tqdm(dataloader):
        images = [img.to(device) for img in images]

        targets = [
            {
                "boxes": t['boxes'].clone().detach().to(device),
                "labels": t['labels'].clone().detach().to(device)
            }
            for t in targets
        ]

        if len(images) == 0 or len(targets) == 0:
            continue

        with torch.set_grad_enabled(training):
            optimizer.zero_grad()
            outputs_class, outputs_bbox = model(torch.stack(images).to(device))

            loss = compute_loss(outputs_class, outputs_bbox, targets)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device):
    """Evaluate the model on the test set and calculate metrics."""
    model.eval()
    pred_boxes_list = []
    pred_labels_list = []
    target_boxes_list = []
    target_labels_list = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [
                {
                    "boxes": t['boxes'].clone().detach().to(device),
                    "labels": t['labels'].clone().detach().to(device)
                }
                for t in targets
            ]

            outputs_class, outputs_bbox = model(torch.stack(images).to(device))

            # Get predicted labels and boxes with confidence filtering
            pred_probs = outputs_class.softmax(-1)
            pred_scores, pred_labels = pred_probs.max(-1)

            # Filter predictions by confidence threshold
            keep = pred_scores > conf_threshold
            pred_labels = pred_labels[keep]
            pred_boxes = outputs_bbox[keep]

            if pred_labels.numel() > 0:
                pred_labels_list.append(pred_labels.cpu())
            if targets:
                target_labels_list.extend([t['labels'].cpu() for t in targets])

            if pred_boxes.numel() > 0:
                pred_boxes_list.append(pred_boxes.cpu())
            if targets:
                target_boxes_list.extend([t['boxes'].cpu() for t in targets])

    # Calculate metrics
    metrics = calculate_metrics(pred_boxes_list, pred_labels_list, target_boxes_list, target_labels_list)
    return metrics
