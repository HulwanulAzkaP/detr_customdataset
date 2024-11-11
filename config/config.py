# config/config.py

import torch

class Config:
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Paths
    train_data_path = 'dataset/train/'
    annotations_train = 'dataset/train/_annotations.coco.json'
    valid_data_path = 'dataset/valid/'
    annotations_valid = 'dataset/valid/_annotations.coco.json'
    test_data_path = 'dataset/test/'
    annotations_test = 'dataset/test/_annotations.coco.json'


    # Training hyperparameters
    num_epochs = 20
    batch_size = 16
    learning_rate = 1e-4

    # Model parameters
    num_classes = 3  # Number of object classes + background
    num_queries = 100
    hidden_dim = 256
    nheads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6

    # Confidence threshold for evaluation
    conf_threshold = 0.5