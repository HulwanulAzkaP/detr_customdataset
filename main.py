import argparse
import torch
from models.detr_model import DETRModel
from data.dataloader import get_dataloader
from training.train import train_one_epoch, evaluate_model
from PIL import Image, ImageDraw, ImageFont
import torch.optim as optim
from config.config import Config
import os
import csv
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for training, evaluation, inference, or prediction with DETRModel.")
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'eval', 'predict'], required=True,
                        help="Mode to run the script: train, inference, eval, or predict")
    return parser.parse_args()


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
        checkpoint = torch.load(filepath, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {filepath} (epoch {checkpoint['epoch']})")
        return checkpoint['epoch']
    return 0


def save_evaluation_metrics(epoch, metrics, filepath="evaluation_metrics.csv"):
    """Save evaluation metrics to a CSV file."""
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Epoch", "Precision", "Recall", "F1-Score"])
        writer.writerow([epoch, metrics['precision'], metrics['recall'], metrics['f1_score']])


def draw_predictions_on_image(image, boxes, labels, scores, output_path):
    """Draw bounding boxes and labels on the image and save it."""
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        box = box.cpu().numpy()
        label = label.item()
        score = score.item()

        # Define box coordinates
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Draw label and confidence score
        text = f"Class {label} ({score:.2f})"
        draw.text((x1, y1), text, fill="white")

    image.save(output_path)
    print(f"Saved prediction to {output_path}")


def predict(model, dataloader, device):
    """Run predictions on the test dataset and save labeled images."""
    model.eval()
    with torch.no_grad():
        for idx, (images, _) in enumerate(dataloader):
            for i, image in enumerate(images):
                image = image.to(device)
                outputs_class, outputs_bbox = model(image.unsqueeze(0))

                # Apply softmax to get class probabilities
                class_probs = F.softmax(outputs_class, dim=-1)

                # Confidence threshold
                confidence_threshold = 0.5
                max_probs, labels = class_probs.squeeze(0).max(dim=-1)
                keep = max_probs > confidence_threshold

                filtered_labels = labels[keep]
                filtered_probs = max_probs[keep]
                filtered_bboxes = outputs_bbox.squeeze(0)[keep]

                # Convert normalized boxes to original image dimensions
                original_image = Image.open(f"{Config.test_data_path}/{dataloader.dataset.images[idx]['file_name']}")
                w, h = original_image.size
                scaled_boxes = filtered_bboxes * torch.tensor([w, h, w, h], device=device)

                # Draw predictions and save image
                output_path = f"predictions/predicted_image_{idx}_{i}.png"
                draw_predictions_on_image(original_image, scaled_boxes, filtered_labels, filtered_probs, output_path)


if __name__ == "__main__":
    args = parse_args()
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

    if args.mode == 'train':
        train_loader = get_dataloader(Config.train_data_path, Config.annotations_train, Config.batch_size)
        valid_loader = get_dataloader(Config.valid_data_path, Config.annotations_valid, Config.batch_size)
        optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        checkpoint_path = "model_checkpoint.pth"
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)

        best_valid_loss = float('inf')
        patience = Config.patience
        patience_counter = 0

        for epoch in range(start_epoch, Config.num_epochs):
            print(f"\nStarting Epoch {epoch + 1}/{Config.num_epochs}")

            train_loss = train_one_epoch(model, train_loader, optimizer, device, training=True)
            print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}")

            print("\nEvaluating on validation set...")
            valid_loss = train_one_epoch(model, valid_loader, optimizer, device, training=False)
            print(f"Epoch {epoch + 1}, Validation Loss: {valid_loss:.4f}")

            if (epoch + 1) % 10 == 0:
                save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)
                print("Validation loss improved, saving model.")
            else:
                patience_counter += 1
                print(f"Validation loss did not improve for {patience_counter} epochs.")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

            metrics = evaluate_model(model, valid_loader, device)
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            save_evaluation_metrics(epoch + 1, metrics)

        print("\nTraining and validation completed!")

    elif args.mode == 'eval':
        valid_loader = get_dataloader(Config.valid_data_path, Config.annotations_valid, Config.batch_size)
        metrics = evaluate_model(model, valid_loader, device)
        print(
            f"Validation Metrics: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1-Score={metrics['f1_score']:.4f}")

    elif args.mode == 'inference':
        test_loader = get_dataloader(Config.test_data_path, Config.annotations_test, Config.batch_size)
        inference(model, test_loader, device)

    elif args.mode == 'predict':
        # Ensure output directory exists
        os.makedirs("predictions", exist_ok=True)

        test_loader = get_dataloader(Config.test_data_path, Config.annotations_test, 1)
        predict(model, test_loader, device)
