import torch
from tqdm import tqdm
from training.loss import compute_loss

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in tqdm(dataloader):
        # Move images to the device
        images = [img.to(device) for img in images]

        # Move targets to the device and fix the warning
        targets = [
            {
                "boxes": t['boxes'].clone().detach().to(device),
                "labels": t['labels'].clone().detach().to(device)
            }
            for t in targets
        ]

        optimizer.zero_grad()
        outputs_class, outputs_bbox = model(torch.stack(images).to(device))

        # Compute loss
        loss = compute_loss(outputs_class, outputs_bbox, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
