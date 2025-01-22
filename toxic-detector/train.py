import pytorch_lightning as pl
import torch
from classifiers import ToxicCommentClassifier
from dataset import get_dataloaders


def train_model():
    train_loader, val_loader, test_loader = get_dataloaders()
    model = ToxicCommentClassifier()

    trainer = pl.Trainer(max_epochs=3, gpus=1 if torch.cuda.is_available() else 0)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    train_model()
