import glob
import os.path

import torchvision.transforms
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class LabeledDataset(Dataset):
    def __init__(self, dataset_dirs: list, filename2labels: dict):
        self.preprocess = torchvision.transforms.Compose(
            [
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4866, 0.4409), (0.2165, 0.2165, 0.2165)
                ),
            ]
        )
        self.images = []
        self.labels = []

        for dataset_dir in dataset_dirs:
            raw_image_paths = glob.glob(
                os.path.join(dataset_dir, "**", "*.png"), recursive=True
            )
            for raw_image_path in raw_image_paths:
                self.images.append(raw_image_path)
                label = int(filename2labels[os.path.basename(raw_image_path)])
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]
        return self.preprocess(image), label


class TestDataset(Dataset):
    def __init__(self, dataset_dirs: list):
        self.preprocess = torchvision.transforms.Compose(
            [
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4866, 0.4409), (0.2165, 0.2165, 0.2165)
                ),
            ]
        )
        self.images = []

        for dataset_dir in dataset_dirs:
            raw_image_paths = glob.glob(
                os.path.join(dataset_dir, "**", "*.png"), recursive=True
            )
            for raw_image_path in raw_image_paths:
                self.images.append(raw_image_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        return self.preprocess(image), {}
