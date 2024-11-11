import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def compute_loss(outputs_class, outputs_bbox, targets):
    batch_size, num_queries, num_classes_plus_one = outputs_class.shape
    pred_classes = outputs_class.view(batch_size * num_queries, num_classes_plus_one)
    pred_boxes = outputs_bbox.view(batch_size * num_queries, 4)

    # Extract target classes and boxes
    target_classes = torch.cat([t['labels'] for t in targets])
    target_boxes = torch.cat([t['boxes'] for t in targets])

    # Perform matching between predicted and target boxes
    row_indices, col_indices = match_predictions(pred_boxes, target_boxes)

    # Gather matched predictions and targets
    matched_pred_classes = pred_classes[row_indices]
    matched_target_classes = target_classes[col_indices]

    matched_pred_boxes = pred_boxes[row_indices]
    matched_target_boxes = target_boxes[col_indices]

    # Classification loss
    if matched_pred_classes.shape[0] > 0:
        class_loss = F.cross_entropy(matched_pred_classes, matched_target_classes)
    else:
        class_loss = torch.tensor(0.0, device=pred_classes.device)

    # Bounding box regression loss
    if matched_pred_boxes.shape[0] > 0:
        bbox_loss = F.l1_loss(matched_pred_boxes, matched_target_boxes, reduction='mean')
    else:
        bbox_loss = torch.tensor(0.0, device=pred_boxes.device)

    total_loss = class_loss + bbox_loss
    return total_loss


def match_predictions(pred_boxes, target_boxes):
    """
    Perform Hungarian matching between predicted and target boxes.
    Returns row and column indices of the matched pairs.
    """
    if len(pred_boxes) == 0 or len(target_boxes) == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    cost_matrix = torch.cdist(pred_boxes, target_boxes, p=1).cpu().detach().numpy()
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    return torch.tensor(row_indices, dtype=torch.long), torch.tensor(col_indices, dtype=torch.long)
