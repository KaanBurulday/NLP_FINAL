import os
import pathlib

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
from torch_directml import device as dml_device
import numpy as np


turkish_stopwords = [
    "acaba", "ama", "ancak", "aslında", "az", "bazı", "belki", "biri", "birkaç", "birşey",
    "biz", "bu", "çok", "çünkü", "da", "daha", "de", "defa", "diye", "eğer", "en", "gibi",
    "hem", "hep", "hepsi", "her", "hiç", "ile", "ise", "kez", "ki", "kim", "mı", "mu",
    "mü", "nasıl", "ne", "neden", "nerde", "nerede", "nereye", "niçin", "niye", "o",
    "sanki", "şey", "siz", "şu", "tüm", "ve", "veya", "ya", "yani"
]

def sanitizer_string(text: str | list[str], stop_words: list[str], only_alpha: bool, split_regex: str) -> str:
    word_list = text if isinstance(text, list) else text.split(split_regex)
    sanitized_text = ""
    if only_alpha:
        for word in word_list:
            if word not in stop_words and word.isalpha():
                sanitized_text += word + " "
    else:
        for word in word_list:
            if word not in stop_words:
                sanitized_text += word + " "
    return sanitized_text

# Define the custom dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Load data
def load_data(data_path):
    texts = []
    labels = []
    label_map = {folder: idx for idx, folder in enumerate(os.listdir(data_path))}
    for folder, label in label_map.items():
        folder_path = os.path.join(data_path, folder)
        for file in os.listdir(folder_path):
            with open(os.path.join(folder_path, file), 'r') as f:
                texts.append(sanitizer_string(f.read(), stop_words=turkish_stopwords, only_alpha=True, split_regex=' '))
                labels.append(label)
    return texts, labels, label_map

# Train the model
def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluate the model
def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return np.array(true_labels), np.array(predictions)


def main():
    data_path = f"{pathlib.Path().resolve()}\\makaleler-yazarlar"
    texts, labels, label_map = load_data(data_path)

    # Stratified 10-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=10)
    tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
    device = dml_device()

    all_reports = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels), 1):
        print(f"Processing Fold {fold}...")

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        train_dataset = TextDataset(train_texts, train_labels, tokenizer)
        test_dataset = TextDataset(test_texts, test_labels, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_map))
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        for epoch in range(3):  # Adjust epochs as needed
            train_loss = train_model(model, train_loader, optimizer, device)

        true_labels, predictions = evaluate_model(model, test_loader, device)

        # Generate classification report
        report = classification_report(
            true_labels,
            predictions,
            target_names=list(label_map.keys()),
            zero_division=0,  # Handle undefined metrics gracefully
        )
        print(f"Classification Report for Fold {fold}:\n{report}")
        all_reports.append(report)

    # Optionally, aggregate metrics across all folds
    print("Cross-Validation Completed.")

if __name__ == "__main__":
    main()