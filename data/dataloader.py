from torch.utils.data import DataLoader
from data.dataset import CocoDetection
from config.config import Config

def get_dataloader(data_path, annotations_path, batch_size, num_workers=2):
    dataset = CocoDetection(data_path, annotations_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )
