import glob
import os.path

import torchvision.transforms
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class MaskDataset(Dataset):
    def __init__(self, mask_dirs: list, non_mask_dirs: list):
        self.preprocess = torchvision.transforms.Compose(
            [
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5508, 0.4830, 0.4514), (0.2582, 0.2582, 0.2582))
            ]
        )
        self.images = []
        self.labels = []

        dirs = [mask_dirs, non_mask_dirs]
        for i in range(len(dirs)):
            for data_dir in dirs[i]:
                raw_image_paths = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
                for raw_image_path in raw_image_paths:
                    self.images.append(raw_image_path)
                    self.labels.append(i)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]
        return self.preprocess(image), label
