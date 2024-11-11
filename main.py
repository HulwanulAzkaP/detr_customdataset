import torch
from models.detr_model import DETRModel
from data.dataloader import get_dataloader
from training.train import train_one_epoch
import torch.optim as optim

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Ensure the model is moved to the GPU
    model = DETRModel(num_classes=3).to(device)
    print(f"Model is on device: {next(model.parameters()).device}")

    train_loader = get_dataloader('dataset/train/', 'dataset/train/_annotations.coco.json', batch_size=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}, Loss: {loss}")
