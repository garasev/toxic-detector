import logging

import hydra
import pytorch_lightning as pl
from classifiers import ToxicCommentClassifier
from dataset import get_dataloaders
from omegaconf import DictConfig
from pytorch_lightning.loggers import MLFlowLogger

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="../config", config_name="config")
def train_model(cfg: DictConfig):
    logging.info("Loading data...")
    train_loader, test_loader = get_dataloaders(
        train_file=cfg.data.train_file,
        test_file=cfg.data.test_file,
        test_labels_file=cfg.data.test_labels_file,
    )
    logging.info("Data loaded successfully.")

    model = ToxicCommentClassifier(
        model_name=cfg.model.model_name,
        num_labels=cfg.model.num_labels,
        lr=cfg.model.lr,
    )
    logging.info("Model initialized.")

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name, tracking_uri=cfg.mlflow.tracking_uri
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs, accelerator="auto", logger=mlflow_logger
    )
    logging.info("Trainer initialized.")

    logging.info("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    logging.info("Training completed.")

    logging.info("Starting testing...")
    trainer.test(model, test_dataloaders=test_loader)
    logging.info("Testing completed.")


if __name__ == "__main__":
    train_model()
