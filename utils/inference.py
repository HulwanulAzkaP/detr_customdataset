import torch
from PIL import Image
from config.config import Config
from models.detr_model import DETRModel

def run_inference(image_path, model, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    inputs = model.image_processor(image, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(inputs)
    return outputs
