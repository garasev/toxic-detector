import torch
from classifiers import ToxicCommentClassifier
from transformers import BertTokenizer


def infer(model_path, text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = ToxicCommentClassifier.load_from_checkpoint(model_path)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.sigmoid(logits)
    return probabilities.item()


if __name__ == "__main__":
    model_path = "path/to/your/model.ckpt"
    text = "This is a sample comment."
    probability = infer(model_path, text)
    print(f"Toxicity probability: {probability}")
