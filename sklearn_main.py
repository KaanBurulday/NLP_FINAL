import cProfile
import json
import math
import os
import pathlib
import pstats
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from pandas import DataFrame
from TextSanitizer import sanitizer_string
from sklearn.metrics import precision_recall_fscore_support

assignment_data_path = f"{pathlib.Path().resolve()}\\{os.getenv('DATA_FOLDER_NAME')}"
tf_idf_table_to_path = f"{pathlib.Path().resolve()}\\TF_IDF_V2_Table.parquet"
results_path = f"{pathlib.Path().resolve()}\\results\\Sklearn_Results.csv"
folds_path_json = f"{pathlib.Path().resolve()}\\folds.json"
folds_exist = True
if folds_exist and not os.path.isfile(path=folds_path_json):
    folds_exist = False
use_tfidf_vectorizer = False
if not use_tfidf_vectorizer and not os.path.isfile(path=tf_idf_table_to_path):
    use_tfidf_vectorizer = True


turkish_stopwords = [
    "acaba", "ama", "ancak", "aslında", "az", "bazı", "belki", "biri", "birkaç", "birşey",
    "biz", "bu", "çok", "çünkü", "da", "daha", "de", "defa", "diye", "eğer", "en", "gibi",
    "hem", "hep", "hepsi", "her", "hiç", "ile", "ise", "kez", "ki", "kim", "mı", "mu",
    "mü", "nasıl", "ne", "neden", "nerde", "nerede", "nereye", "niçin", "niye", "o",
    "sanki", "şey", "siz", "şu", "tüm", "ve", "veya", "ya", "yani"
]
use_custom_tfidf_config = False
TF_IDF_config = {
    "max_features": 15000,
    "min_df": 0.005,
    "max_df": 0.7,
}
stratified_kfolds_config = {
    'K': 10,  # Fold Amount
}


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
                        data.append({'text': text, 'class': folder_path.name})
    return DataFrame(data)


def get_folds_json(path):
    folds_df = {}
    with open(path, "r") as file:
        folds = json.load(file)
    for key, value in folds.items():
        folds_df[key] = pd.DataFrame(value)

    return folds_df


# Custom KNN Classifier with Cosine Similarity
class KNNWithCosineSimilarity:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.knn = None
        self.labels = None

    def fit(self, X, y):
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine')
        self.knn.fit(X)
        self.labels = y

    def predict(self, X):
        distances, indices = self.knn.kneighbors(X)
        predictions = []
        for idx_list in indices:
            neighbor_labels = self.labels[idx_list]
            most_common = pd.Series(neighbor_labels).mode()[0]
            predictions.append(most_common)
        return np.array(predictions)

def calculate_metrics_multiclass(y_true, y_pred):
    """
    Calculate Precision, Recall, and F1-Score for multi-class classification.

    Parameters:
        y_true (list): List of actual class labels.
        y_pred (list): List of predicted class labels.

    Returns:
        dict: Dictionary containing Precision, Recall, and F1-Score for each class and their averages.
    """
    metrics = {}
    classes = np.unique(y_true)
    num_classes = len(classes)
    total_precision, total_recall, total_f1 = 0, 0, 0

    for c in classes:
        # Calculate TP, FP, FN for class c
        TP = sum((yt == c and yp == c) for yt, yp in zip(y_true, y_pred))
        FP = sum((yt != c and yp == c) for yt, yp in zip(y_true, y_pred))
        FN = sum((yt == c and yp != c) for yt, yp in zip(y_true, y_pred))

        # Precision, Recall, F1-Score for class c
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Store metrics for this class
        metrics[f"{c}"] = {
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_score
        }

        # Aggregate for averaging
        total_precision += precision
        total_recall += recall
        total_f1 += f1_score

    # Calculate macro-averaged metrics
    avg_precision = total_precision / num_classes
    avg_recall = total_recall / num_classes
    avg_f1_score = total_f1 / num_classes

    metrics["Average"] = {
        "Precision": avg_precision,
        "Recall": avg_recall,
        "F1-Score": avg_f1_score
    }

    return metrics

def sklearn_main():
    # Load and preprocess data
    only_alpha = True
    split_regex = " "

    # Get the data
    data = get_data(assignment_data_path, turkish_stopwords, only_alpha, split_regex)

    # Extract features and labels
    texts = data['text']
    labels = data['class']
    knn_neighbors = int(math.sqrt(data.shape[0]))

    # Initialize TF-IDF Vectorizer and fit once on the entire dataset
    if use_tfidf_vectorizer:
        if use_custom_tfidf_config:
            vectorizer = TfidfVectorizer(max_features=TF_IDF_config["max_features"],
                                         min_df=TF_IDF_config["min_df"],
                                         max_df=TF_IDF_config["max_df"])
        else:
            vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts).toarray()
    else:
        tfidf_data = pd.read_parquet(tf_idf_table_to_path)
        X = tfidf_data.drop(columns=['class']).values
    y = labels.values

    predicted_labels = []
    actual_labels = []

    if not folds_exist:
        # Stratified K-Fold Cross Validation
        skf = StratifiedKFold(n_splits=stratified_kfolds_config['K'], shuffle=True, random_state=42)

        for train_index, test_index in skf.split(X, y):
            # Use the precomputed TF-IDF matrix
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train KNN with cosine similarity
            knn = KNNWithCosineSimilarity(n_neighbors=knn_neighbors)
            knn.fit(X_train, y_train)

            # Predict on test data
            y_pred = knn.predict(X_test)

            predicted_labels.extend(y_pred)
            actual_labels.extend(y_test)
    else:
        folds_df = get_folds_json(path=folds_path_json)

        # Perform K-Fold Cross-Validation using folds.json
        for fold, test_data in folds_df.items():
            # Split data into train/test sets
            train_data = pd.concat([data for fold_x, data in folds_df.items() if fold_x != fold])

            if train_data.empty or test_data.empty:
                print(f"Fold {fold}: train_data or test_data is empty. Skipping this fold.")
                continue

            # Use the precomputed TF-IDF matrix for train and test subsets
            train_indices = train_data.index.to_numpy().astype(int)
            test_indices = test_data.index.to_numpy().astype(int)
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_test = X[test_indices]
            y_test = y[test_indices]

            # Train KNN with cosine similarity
            knn = KNNWithCosineSimilarity(n_neighbors=knn_neighbors)
            knn.fit(X_train, y_train)

            # Predict on test data
            y_pred = knn.predict(X_test)

            predicted_labels.extend(y_pred)
            actual_labels.extend(y_test)

    metrics = calculate_metrics_multiclass(y_true=actual_labels, y_pred=predicted_labels)
    print(metrics["Average"])
    pd.DataFrame(metrics).to_csv(results_path)
    return metrics


def sklearn_run_with_profiler():
    start_time = time.time()
    profile_output_filename = 'profile_outputs\\sklearn_profile_output'
    cProfile.run('sklearn_main()', profile_output_filename)
    p = pstats.Stats(profile_output_filename)
    p.sort_stats(pstats.SortKey.TIME).print_stats(10)

    print("--- sklearn total duration: %s seconds ---" % (time.time() - start_time))

# if __name__ == '__main__':
#     sklearn_run_with_profiler()
