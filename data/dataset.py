import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class CocoDetection(Dataset):
    def __init__(self, root, annotation):
        self.root = root
        with open(annotation, 'r') as f:
            self.coco = json.load(f)
        self.images = self.coco['images']
        self.annotations = self.coco['annotations']
        self.transforms = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_info = self.images[index]
        image_path = os.path.join(self.root, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)

        image_id = image_info['id']
        target = [ann for ann in self.annotations if ann['image_id'] == image_id]
        boxes = [ann['bbox'] for ann in target]
        labels = [ann['category_id'] for ann in target]

        return image, {"boxes": boxes, "labels": labels}
