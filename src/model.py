import pytorch_lightning as pl
import torch
import torchmetrics
from torchmetrics import F1Score
from torchvision.models import resnet18, vgg13


class ImageClassifierResnet(torch.nn.Module):
    def __init__(self, lr: float, num_classes: int):
        super().__init__()
        self.model = resnet18(pretrained=True)
        num_last_features = self.resnet.fc.out_features
        self.regressor = torch.nn.Linear(num_last_features, num_classes)
        self.lr = lr

    def forward(self, image):
        outputs = self.model(image)
        logits = self.regressor(outputs)
        return torch.nn.functional.softmax(logits, 1)


class ImageClassifierVGG(torch.nn.Module):
    def __init__(self, lr: float, num_classes):
        super().__init__()
        self.model = vgg13(pretrained=True)
        self.regressor = torch.nn.Linear(1000, num_classes)
        self.lr = lr

    def forward(self, image):
        outputs = self.model(image)
        logits = self.regressor(outputs)
        return torch.nn.functional.softmax(logits, 1)


class TrainModel(pl.LightningModule):
    def __init__(self, lr, model, num_classes):
        super().__init__()
        self.model = model(lr, num_classes)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        output = self.resnet(batch[0])
        cross_entropy = torch.nn.CrossEntropyLoss()
        loss = cross_entropy(output, batch[1])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(batch[0])
        cross_entropy = torch.nn.CrossEntropyLoss()
        loss = cross_entropy(output, batch[1])
        self.log("val_loss", loss)
        predicted = torch.argmax(output, dim=1)
        acc = torchmetrics.Accuracy().to(device="cuda")
        value = acc(predicted, batch[1])
        self.log("val_accuracy", value)
        f1 = F1Score(num_classes=20).to(device="cuda")
        self.log("val_f1", f1(predicted, batch[1]))

    def test_step(self, batch, batch_idx):
        output = self.model(batch[0])
        cross_entropy = torch.nn.CrossEntropyLoss()
        loss = cross_entropy(output, batch[1])
        self.log("test_loss", loss)
        predicted = torch.argmax(output, dim=1)
        acc = torchmetrics.Accuracy(num_classes=20).to(device="cuda")
        value = acc(predicted, batch[1])
        self.log("test_accuracy", value)
        f1 = F1Score(num_classes=20).to(device="cuda")
        self.log("test_f1", f1(predicted, batch[1]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
