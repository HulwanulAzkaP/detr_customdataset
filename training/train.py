import torch
from tqdm import tqdm
from training.loss import compute_loss


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in tqdm(dataloader):
        # Move images and targets to the device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Check device placement
        print(f"Images on device: {images[0].device}")
        print(f"Targets on device: {targets[0]['boxes'].device}")

        optimizer.zero_grad()
        outputs_class, outputs_bbox = model(torch.stack(images).to(device))

        # Compute loss
        loss = compute_loss(outputs_class, outputs_bbox, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
