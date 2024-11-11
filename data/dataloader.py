from torch.utils.data import DataLoader
from data.dataset import CocoDetection

def get_dataloader(data_path, annotations_path, batch_size):
    dataset = CocoDetection(data_path, annotations_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
