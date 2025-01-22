import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


class ToxicCommentDataset(Dataset):
    def __init__(self, tokenized_data, labels):
        self.data = tokenized_data
        self.labels = labels

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.data.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def get_dataloaders(
    batch_size=16,
    train_file="data/train.csv",
    test_file="data/test.csv",
    test_labels_file="data/test_labels.csv",
):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def load_csv(file_path):
        df = pd.read_csv(file_path)
        if "comment_text" not in df.columns:
            raise KeyError(f"Column 'comment_text' not found in {file_path}")
        df["comment_text"] = df["comment_text"].astype(str)
        return df

    def tokenize_function(examples):
        return tokenizer(
            examples["comment_text"].tolist(),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    train_df = load_csv(train_file)
    test_df = load_csv(test_file)
    test_labels_df = pd.read_csv(test_labels_file)

    train_tokenized = tokenize_function(train_df)
    test_tokenized = tokenize_function(test_df)

    train_labels = train_df[
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    ].values
    test_labels = test_labels_df[
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    ].values

    train_dataset = ToxicCommentDataset(train_tokenized, train_labels)
    test_dataset = ToxicCommentDataset(test_tokenized, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader
