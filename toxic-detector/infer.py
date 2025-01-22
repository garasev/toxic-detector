import logging

import hydra
import mlflow
import pandas as pd
import torch
from classifiers import ToxicCommentClassifier
from omegaconf import DictConfig
from transformers import BertTokenizer

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="../config", config_name="config")
def infer(cfg: DictConfig):
    logging.info("Loading model...")
    tokenizer = BertTokenizer.from_pretrained(cfg.model.model_name)
    model = ToxicCommentClassifier.load_from_checkpoint(
        cfg.model.model_path, num_labels=cfg.model.num_labels
    )
    model.eval()

    test_df = pd.read_csv(cfg.data.test_file)
    if "comment_text" not in test_df.columns:
        raise KeyError(f"Column 'comment_text' not found in {cfg.data.test_file}")
    test_texts = test_df["comment_text"].tolist()

    test_labels_df = pd.read_csv(cfg.data.test_labels_file)
    test_ids = test_labels_df["id"].tolist()

    inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.sigmoid(logits)

    predictions = probabilities.cpu().numpy()
    submission_df = pd.DataFrame({"id": test_ids})
    submission_df[
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    ] = predictions
    submission_df.to_csv(cfg.data.output_file, index=False)

    mlflow.log_artifact(cfg.data.output_file)


if __name__ == "__main__":
    infer()
