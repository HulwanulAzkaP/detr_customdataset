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
    indices = match_predictions(pred_boxes, target_boxes)

    # Gather matched predictions
    matched_pred_classes = pred_classes[indices]
    matched_target_classes = target_classes

    # Classification loss
    class_loss = F.cross_entropy(matched_pred_classes, matched_target_classes)

    # Bounding box regression loss
    matched_pred_boxes = pred_boxes[indices]
    bbox_loss = F.l1_loss(matched_pred_boxes, target_boxes, reduction='mean')

    total_loss = class_loss + bbox_loss
    return total_loss


def match_predictions(pred_boxes, target_boxes):
    cost_matrix = torch.cdist(pred_boxes, target_boxes, p=1).cpu().detach().numpy()
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    indices = row_indices
    return torch.tensor(indices, dtype=torch.long)
