import torch

class Config:
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42

    # Model and training settings
    checkpoint = 'facebook/detr-resnet-50'
    confidence_threshold = 0.5
    iou_threshold = 0.8
    batch_size = 4
    num_workers = 4
    lr = 1e-4
    epochs = 1
    num_classes = 3  # Update this based on your dataset
    image_size = (640, 640)

    # Dataset paths
    train_data_path = 'dataset/train/'
    valid_data_path = 'dataset/valid/'
    test_data_path = 'dataset/test/'
    annotations_train = 'dataset/train/_annotations.coco.json'
    annotations_valid = 'dataset/valid/_annotations.coco.json'
    annotations_test = 'dataset/test/_annotations.coco.json'

    # Checkpoints
    checkpoint_dir = 'checkpoints/'
    model_save_path = 'checkpoints/detr_model.pth'
