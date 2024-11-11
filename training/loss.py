import torch
import torch.nn.functional as F

def compute_loss(outputs_class, outputs_bbox, targets):
    # Classification loss
    target_classes = torch.cat([t['labels'] for t in targets])
    pred_classes = outputs_class.view(-1, outputs_class.shape[-1])
    class_loss = F.cross_entropy(pred_classes, target_classes)

    # Bounding box loss
    target_boxes = torch.cat([t['boxes'] for t in targets])
    pred_boxes = outputs_bbox.view(-1, 4)
    bbox_loss = F.l1_loss(pred_boxes, target_boxes)

    total_loss = class_loss + bbox_loss
    return total_loss
