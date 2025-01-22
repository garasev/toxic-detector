import pytorch_lightning as pl
import torch
from transformers import BertForSequenceClassification


class ToxicCommentClassifier(pl.LightningModule):
    def __init__(self, model_name="bert-base-uncased", lr=2e-5):
        super(ToxicCommentClassifier, self).__init__()
        self.save_hyperparameters()
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=1
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
