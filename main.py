import csv
import datetime
import os.path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import conf
import src.dataset_builder
from src.model import TrainModel

filepath2label = {}

LOG_NAME = conf.LOG_NAME

with open("train_master.tsv", mode="r") as fh:
    reader = csv.reader(fh, delimiter="\t")
    filepath2label = {row[0]: row[1] for row in reader}


dataset = src.dataset_builder.LabeledDataset(conf.DATA_SET, filepath2label)

n_samples = len(dataset)
train_size = int(len(dataset) * 0.6)
val_size = int(len(dataset) * 0.2)
test_size = n_samples - (train_size + val_size)

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32 * conf.BATCH_SIZE,
    num_workers=os.cpu_count(),
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32 * conf.BATCH_SIZE, num_workers=os.cpu_count()
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32 * conf.BATCH_SIZE, num_workers=os.cpu_count()
)

checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_weights_only=True,
    dirpath="model/",
    filename=f"{str(datetime.datetime.today())}",
)
lr_monitor = LearningRateMonitor(logging_interval="epoch")

logger = TensorBoardLogger(save_dir=".", version=LOG_NAME)

trainer = pl.Trainer(
    logger=logger,
    gpus=1,
    max_epochs=conf.MAX_EPOCHS,
    callbacks=[
        checkpoint,
        pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", mode="min"),
        lr_monitor,
    ],
)

model = TrainModel(lr=1e-5, num_classes=conf.NUM_CLASSES)
trainer.fit(model, train_loader, val_loader)

test = trainer.test(model, test_loader)
print(test)
