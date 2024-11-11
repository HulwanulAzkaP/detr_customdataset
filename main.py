import torch
from models.detr_model import DETRModel
from data.dataloader import get_dataloader
from training.train import train_one_epoch
import torch.optim as optim
from config.config import Config

if __name__ == "__main__":
    device = Config.device
    print(f"Using device: {device}")

    # Initialize the model
    model = DETRModel(
        num_classes=Config.num_classes,
        num_queries=Config.num_queries,
        hidden_dim=Config.hidden_dim,
        nheads=Config.nheads,
        num_encoder_layers=Config.num_encoder_layers,
        num_decoder_layers=Config.num_decoder_layers
    ).to(device)
    print(f"Model is on device: {next(model.parameters()).device}")

    # Get DataLoader
    train_loader = get_dataloader(Config.train_data_path, Config.annotations_train, Config.batch_size)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

    for epoch in range(Config.num_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}, Loss: {loss}")
