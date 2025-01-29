import cProfile
import os
import pathlib
import pstats
import time

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold
from transformers import DataCollatorWithPadding, RobertaTokenizer, RobertaForSequenceClassification

from UnimportedTools import Notifier
from TextSanitizer import sanitizer_string
import torch
from torch.utils.data import DataLoader, Dataset
from torch_directml import device as dml_device
from torch.optim import AdamW


def get_data(data_base_path, stop_words, only_alpha, split_regex):
    data = []
    base_path = pathlib.Path(data_base_path)
    for folder_path in base_path.iterdir():
        if folder_path.is_dir():
            for file_path in folder_path.iterdir():
                if file_path.is_file():
                    with file_path.open('r') as f:
                        text = sanitizer_string(
                            text=f.read(),
                            stop_words=stop_words,
                            only_alpha=only_alpha,
                            split_regex=split_regex
                        )
                        data.append({'text': text, 'labels': folder_path.name})
    return pd.DataFrame(data)

save_models = True
model_name = "burakaytan/roberta-base-turkish-uncased"  # roberta-base, roberta-large (use batch_size 16),
# TURKCELL/roberta-base-turkish-uncased, burakaytan/roberta-base-turkish-uncased
batch_size = 16
max_length = 256
learning_rate = 3e-5
model_file_name = (model_name.replace("-", "_")
                   .replace(".", "_")
                   .replace("/", "_")) + f"{max_length}_" + f"{batch_size}"

device = dml_device()
assignment_data_path = f"{pathlib.Path().resolve()}\\{os.getenv('DATA_FOLDER_NAME')}"
print(f"Using data: {os.getenv('DATA_FOLDER_NAME')}")
only_alpha = True
split_regex = " "
turkish_stopwords = [
    "acaba", "ama", "ancak", "aslında", "az", "bazı", "belki", "biri", "birkaç", "birşey",
    "biz", "bu", "çok", "çünkü", "da", "daha", "de", "defa", "diye", "eğer", "en", "gibi",
    "hem", "hep", "hepsi", "her", "hiç", "ile", "ise", "kez", "ki", "kim", "mı", "mu",
    "mü", "nasıl", "ne", "neden", "nerde", "nerede", "nereye", "niçin", "niye", "o",
    "sanki", "şey", "siz", "şu", "tüm", "ve", "veya", "ya", "yani"
]
data = get_data(assignment_data_path, turkish_stopwords, only_alpha, split_regex)
print(data.head())
data['encoded_cat'] = data['labels'].astype('category').cat.codes

labels = data['labels']
encoded_labels = data['encoded_cat']
unique_encoded_labels = set(encoded_labels)
unique_labels = set(labels)
num_labels = len(unique_labels)
unique_pairs = data[['labels', 'encoded_cat']].drop_duplicates().to_numpy()

label_to_int = {row[1]: row[0] for row in unique_pairs}

X = data["text"]
y = data["encoded_cat"]

all_predictions = []
all_true_labels = []


def run_RoBERTa():
    global predictions, true_labels
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer):
            self.labels = labels
            self.encodings = tokenizer(
                texts,
                padding="longest",  # Dynamic padding works better when used with a DataCollator later
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    model = RobertaForSequenceClassification.from_pretrained(model_name,
                                                             num_labels=num_labels)
    model.to(device)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Starting Fold {fold + 1}")
        fold_start_time = time.time()

        train_texts, val_texts = X.iloc[train_idx].tolist(), X.iloc[val_idx].tolist()
        train_labels, val_labels = y.iloc[train_idx].tolist(), y.iloc[val_idx].tolist()

        train_dataset = TextDataset(train_texts, train_labels, tokenizer)
        val_dataset = TextDataset(val_texts, val_labels, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator)

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        # optimizer = SGD(model.parameters(), lr=learning_rate)

        # Early stopping parameters
        best_val_loss = float("inf")
        early_stop_counter = 0
        patience = 3  # Number of epochs to wait before stopping
        max_epochs = 10  # Maximum number of epochs for training
        threshold = 1e-4  # Minimum improvement required

        # Training and validation loop
        for epoch in range(max_epochs):
            print(f"Epoch {epoch + 1}/{max_epochs}")
            model.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            print(f"Fold {fold + 1} | Epoch {epoch + 1} | Train Loss: {train_loss:.4f}")

            # Evaluate on validation set
            model.eval()
            val_loss = 0
            predictions = []
            true_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    val_loss += outputs.loss.item()

                    preds = torch.argmax(outputs.logits, dim=1)
                    predictions.extend(preds.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())

            print(f"Fold {fold + 1} | Epoch {epoch + 1} | Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                if improvement < threshold:
                    print(
                        f"Fold {fold + 1} | Epoch {epoch + 1} | Improvement ({improvement:.6f}) below threshold ({threshold}). Early stopping triggered.")
                    break
                best_val_loss = val_loss
                early_stop_counter = 0
                # Save the best model for this fold
                if save_models:
                    torch.save(model.state_dict(), f"RoBERTa_Models\\roberta_{model_file_name}_best_model_fold_{fold + 1}.pt")
                print(
                    f"Fold {fold + 1} | Epoch {epoch + 1} | New Best Model Saved with Improvement: {improvement:.6f}!")
            else:
                early_stop_counter += 1
                print(f"Fold {fold + 1} | Epoch {epoch + 1} | No Improvement. Early Stop Counter: {early_stop_counter}")

                if early_stop_counter >= patience:
                    print(
                        f"Fold {fold + 1} | Early stopping triggered due to no improvement for {patience} consecutive epochs.")
                    break

            # Track predictions and true labels for overall metrics
        all_predictions.extend(predictions)
        all_true_labels.extend(true_labels)

        print(f"--- Fold {fold + 1} duration: %s seconds ---" % (time.time() - fold_start_time))

    results = calculate_metrics_multiclass(y_true=all_true_labels, y_pred=all_predictions)
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{pathlib.Path().resolve()}\\results\\RoBERTa_{model_file_name}_Results.csv", index=False)
    print(results_df)

def evaluate_with_loaded_models():
    """
    Loads the trained BERT models for each fold and evaluates on the dataset using stratified 10-fold CV.
    """
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer):
            self.labels = labels
            self.encodings = tokenizer(
                texts,
                padding="longest",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    all_predictions = []
    all_true_labels = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Evaluating Fold {fold + 1}")
        fold_start_time = time.time()

        val_texts = X.iloc[val_idx].tolist()
        val_labels = y.iloc[val_idx].tolist()

        val_dataset = TextDataset(val_texts, val_labels, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Load saved model
        model_path = f"RoBERTa_Models\\roberta_{model_file_name}_best_model_fold_{fold + 1}.pt"
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found! Skipping Fold {fold + 1}.")
            continue

        model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        predictions = []
        true_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        all_predictions.extend(predictions)
        all_true_labels.extend(true_labels)

        print(f"--- Fold {fold + 1} evaluation duration: %s seconds ---" % (time.time() - fold_start_time))

    # Compute metrics
    results = calculate_metrics_multiclass(y_true=all_true_labels, y_pred=all_predictions)
    print(f"calculate_metrics_multiclass: {results}")
    # results_df = pd.DataFrame(results)
    # results_df.to_csv(f"{pathlib.Path().resolve()}\\results\\BERT_{model_file_name}_Evaluation_Results.csv", index=False)

    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_predictions, average="weighted")
    accuracy = accuracy_score(all_true_labels, all_predictions)

    results = {
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1-score": [f1]
    }

    results_df = pd.DataFrame(results)
    results_file = f"{pathlib.Path().resolve()}\\results\\BERT_{model_file_name}_Evaluation_Results.csv"
    results_df.to_csv(results_file, index=False)

    print(f"precision_recall_fscore_support: {results}")
    return results



def calculate_metrics_multiclass(y_true, y_pred):
    """Calculate metrics for multi-class classification."""
    metrics = {}
    total_precision, total_recall, total_f1 = 0, 0, 0

    for c in unique_encoded_labels:
        TP = sum((yt == c and yp == c) for yt, yp in zip(y_true, y_pred))
        FP = sum((yt != c and yp == c) for yt, yp in zip(y_true, y_pred))
        FN = sum((yt == c and yp != c) for yt, yp in zip(y_true, y_pred))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics[f"{label_to_int[c]}"] = {"Precision": precision, "Recall": recall, "F1-Score": f1_score}
        total_precision += precision
        total_recall += recall
        total_f1 += f1_score

    metrics["Average"] = {
        "Precision": total_precision / num_labels,
        "Recall": total_recall / num_labels,
        "F1-Score": total_f1 / num_labels,
    }
    print(metrics["Average"])
    return metrics


def roberta_run_with_profiler():
    start_time = time.time()
    profile_output_filename = f'profile_outputs\\roberta_{model_file_name}_profile_output'
    cProfile.run('run_RoBERTa()', profile_output_filename)
    p = pstats.Stats(profile_output_filename)
    p.sort_stats(pstats.SortKey.TIME).print_stats(10)

    Notifier.notify_complete()

    print("--- RoBERTa total duration: %s seconds ---" % (time.time() - start_time))

def roberta_evaluate_run_with_profiler():
    start_time = time.time()
    profile_output_filename = f'profile_outputs\\roberta__evaluate_{model_file_name}_profile_output'
    cProfile.run('evaluate_with_loaded_models()', profile_output_filename)
    p = pstats.Stats(profile_output_filename)
    p.sort_stats(pstats.SortKey.TIME).print_stats(10)

    Notifier.notify_complete()

    print("--- RoBERTa evaluation total duration: %s seconds ---" % (time.time() - start_time))
