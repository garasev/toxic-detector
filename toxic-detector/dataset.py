import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


class ToxicCommentDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.data.items()
            if key != "comment_text"
        }
        item["labels"] = torch.tensor(self.data["toxic"][idx])
        return item


def get_dataloaders(batch_size=16):
    dataset = load_dataset("jigsaw-toxic-comment-classification-challenge")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(
            examples["comment_text"], padding="max_length", truncation=True
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    train_dataset = ToxicCommentDataset(tokenized_datasets["train"])
    val_dataset = ToxicCommentDataset(tokenized_datasets["validation"])
    test_dataset = ToxicCommentDataset(tokenized_datasets["test"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
