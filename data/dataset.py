import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch


class CocoDetection(Dataset):
    def __init__(self, root, annotation):
        self.root = root
        with open(annotation, 'r') as f:
            self.coco = json.load(f)
        self.images = self.coco['images']
        self.annotations = self.coco['annotations']
        self.transforms = T.Compose([T.Resize((512, 512)), T.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_info = self.images[index]
        image_path = os.path.join(self.root, image_info['file_name'])

        # Buka gambar
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)

        image_id = image_info['id']
        target = [ann for ann in self.annotations if ann['image_id'] == image_id]

        if len(target) == 0:
            return image, {"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.int64)}

        # Ambil ukuran gambar
        img_width, img_height = image.size(2), image.size(1)

        # Ekstrak bounding boxes dan lakukan normalisasi
        boxes = [ann['bbox'] for ann in target]
        boxes = torch.tensor(boxes, dtype=torch.float32)
        boxes[:, [0, 2]] /= img_width  # Normalisasi x dan width
        boxes[:, [1, 3]] /= img_height  # Normalisasi y dan height

        # Ambil label
        labels = [ann['category_id'] for ann in target]
        labels = torch.tensor(labels, dtype=torch.int64)

        return image, {"boxes": boxes, "labels": labels}
