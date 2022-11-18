import torch
import pytorch_lightning as pl
import torchmetrics
from torchvision.models import resnet18


class Model(torch.nn.Module):
    def __init__(self, lr):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        num_last_features = self.resnet.fc.out_features
        self.regressor = torch.nn.Linear(num_last_features, 2)
        self.lr = lr

    def forward(self, image):
        outputs = self.resnet(image)
        logits = self.regressor(outputs)
        return torch.nn.functional.softmax(logits, 1)


class TrainModel(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.model = Model(lr)
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

    def test_step(self, batch, batch_idx):
        output = self.resnet(batch[0])
        cross_entropy = torch.nn.CrossEntropyLoss()
        loss = cross_entropy(output, batch[1])
        self.log("test_loss", loss)
        predicted = torch.argmax(output,dim=1)
        acc = torchmetrics.Accuracy().to(device="cuda")
        value = acc(predicted, batch[1])
        self.log("test_accuracy", value)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
