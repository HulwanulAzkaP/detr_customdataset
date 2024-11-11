import torch
from models.detr_model import DETRModel
from data.dataloader import get_dataloader
from training.train import train_one_epoch, evaluate_model
import torch.optim as optim
from config.config import Config
import os


def save_checkpoint(model, optimizer, epoch, filepath):
    """Save the model and optimizer state."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """Load the model and optimizer state."""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {filepath} (epoch {checkpoint['epoch']})")
        return checkpoint['epoch']
    return 0


if __name__ == "__main__":
    device = Config.device
    print(f"Using device: {device}")

    model = DETRModel(
        num_classes=Config.num_classes,
        num_queries=Config.num_queries,
        hidden_dim=Config.hidden_dim,
        nheads=Config.nheads,
        num_encoder_layers=Config.num_encoder_layers,
        num_decoder_layers=Config.num_decoder_layers
    ).to(device)
    print(f"Model is on device: {next(model.parameters()).device}")

    train_loader = get_dataloader(Config.train_data_path, Config.annotations_train, Config.batch_size)
    valid_loader = get_dataloader(Config.valid_data_path, Config.annotations_valid, Config.batch_size)
    test_loader = get_dataloader(Config.test_data_path, Config.annotations_test, Config.batch_size)

    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    checkpoint_path = "model_checkpoint.pth"
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)

    for epoch in range(start_epoch, Config.num_epochs):
        print(f"\nStarting Epoch {epoch + 1}/{Config.num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, training=True)
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)

        print("\nEvaluating on validation set...")
        valid_loss = train_one_epoch(model, valid_loader, optimizer, device, training=False)
        print(f"Epoch {epoch + 1}, Validation Loss: {valid_loss:.4f}")

    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, test_loader, device)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")

    print("\nTraining and evaluation completed!")
