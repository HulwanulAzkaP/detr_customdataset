import torch
from tqdm import tqdm
from config.config import Config

def evaluate(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_images = 0
        for images, targets in tqdm(dataloader):
            images = [img.to(device) for img in images]
            outputs = model(images)
            # Evaluation logic can be added here
            total_images += len(images)
    print(f"Total Images: {total_images}")
