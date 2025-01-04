import pathlib


from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import os
import pandas as pd

data = []
base_path = f"{pathlib.Path().resolve()}\\makaleler-yazarlar"

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

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)


for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as f:
                text = sanitizer_string(text=f.read(), stop_words=turkish_stopwords, only_alpha=True, split_regex=' ')
                data.append({'text': text, 'label': folder})

df = pd.DataFrame(data)

print(df)

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=30)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

results = []

for train_index, test_index in skf.split(df['text'], df['label']):
    train_texts = df['text'].iloc[train_index].tolist()
    train_labels = df['label'].iloc[train_index].tolist()
    test_texts = df['text'].iloc[test_index].tolist()
    test_labels = df['label'].iloc[test_index].tolist()

    # Tokenize
    train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    test_encodings = tokenizer(test_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
    test_dataset = torch.utils.data.TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training
    model.train()
    for epoch in range(3):  # 3 epochs
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            preds.extend(torch.argmax(outputs.logits, axis=1).tolist())
            true_labels.extend(labels.tolist())

    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted')
    results.append({'precision': precision, 'recall': recall, 'f1': f1})

# Average results
avg_precision = sum([r['precision'] for r in results]) / 10
avg_recall = sum([r['recall'] for r in results]) / 10
avg_f1 = sum([r['f1'] for r in results]) / 10

print(f'Precision: {avg_precision}, Recall: {avg_recall}, F1: {avg_f1}')

model.save_pretrained(f"{pathlib.Path().resolve()}\\pretrained_model")
tokenizer.save_pretrained(f"{pathlib.Path().resolve()}\\pretrained_tokenizer")
