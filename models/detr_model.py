import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import Transformer
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class DETRModel(nn.Module):
    def __init__(self, num_classes, num_queries=100, hidden_dim=256, nheads=8, num_encoder_layers=6,
                 num_decoder_layers=6):
        super(DETRModel, self).__init__()
        # Backbone: ResNet-50 with updated weights
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Transformer
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True
        )
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Detection heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # Including background class
        self.bbox_embed = nn.Linear(hidden_dim, 4)

    def forward(self, images, targets=None):
        # Extract features using the backbone
        features = self.backbone(images)
        features = self.conv(features)
        bs, c, h, w = features.shape
        features = features.flatten(2).permute(0, 2, 1)  # Shape: (batch_size, seq_len, hidden_dim)

        # Positional encoding
        pos_enc = self.pos_encoder(features)

        # Prepare queries
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1,
                                                                  1)  # Shape: (batch_size, num_queries, hidden_dim)

        # Pass through Transformer
        transformer_out = self.transformer(
            src=pos_enc,  # Shape: (batch_size, seq_len, hidden_dim)
            tgt=query_embed  # Shape: (batch_size, num_queries, hidden_dim)
        )

        # Get class and bounding box predictions
        outputs_class = self.class_embed(transformer_out)
        outputs_bbox = self.bbox_embed(transformer_out).sigmoid()

        return outputs_class, outputs_bbox
