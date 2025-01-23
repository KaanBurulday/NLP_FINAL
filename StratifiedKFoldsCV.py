import pathlib
from collections import Counter

import pandas as pd
from pandas import DataFrame

from FoldCreator import FoldCreator
from TextSanitizer import sanitizer_string
import os
from knn import KNN

turkish_stopwords = [
    "acaba", "ama", "ancak", "aslında", "az", "bazı", "belki", "biri", "birkaç", "birşey",
    "biz", "bu", "çok", "çünkü", "da", "daha", "de", "defa", "diye", "eğer", "en", "gibi",
    "hem", "hep", "hepsi", "her", "hiç", "ile", "ise", "kez", "ki", "kim", "mı", "mu",
    "mü", "nasıl", "ne", "neden", "nerde", "nerede", "nereye", "niçin", "niye", "o",
    "sanki", "şey", "siz", "şu", "tüm", "ve", "veya", "ya", "yani"
]
fold_creator_config = {
    'data_base_path': None,
    'K': 10,
    'stop_words': turkish_stopwords,
    'only_alpha': True,
    'split_regex': ' ',
    'use_sklearn': True
}


class StratifiedKFoldsCV:
    def __init__(self, **kwargs):
        self.K = kwargs.get('K', 10)
        fold_creator_config['K'] = self.K
        self.data_base_path = kwargs.get('data_base_path', None)
        if self.data_base_path is None:
            raise ValueError('data_base_path needs to be provided')
        self.folds_created = kwargs.get('folds_created', False)
        if self.folds_created:
            self.data = kwargs.get('data')
            self.folds = kwargs.get('folds')
            self.num_of_classes = len(self.data['class'].unique())
        else:
            fold_creator_config['data_base_path'] = self.data_base_path
            self.fold_creator_config = kwargs.get('fold_creator_config', fold_creator_config)
            self.fold_creator = FoldCreator(**fold_creator_config)
            self.data = self.fold_creator.data
            self.folds = self.fold_creator.folds
            self.num_of_classes = len(self.data['class'].unique())

        self.knn_config = kwargs.get('knn_config', None)
        if self.knn_config is None:
            raise ValueError('knn_config needs to be provided')

        self.predicted_labels = []
        self.actual_labels = []

    def start_knn(self):
        for fold in self.folds:
            train_folds = {train_fold: self.folds[train_fold] for train_fold in self.folds if train_fold != fold}
            combined_train_data = pd.concat(train_folds.values())
            self.knn_config['train_data'] = combined_train_data.to_numpy()
            knn = KNN(**self.knn_config)
            result = knn.predict_bulk(self.folds[fold].to_numpy())
            self.predicted_labels += result['predictions']
            self.actual_labels += result['actual']

        metrics = self.calculate_metrics_multiclass(y_true=self.actual_labels,
                                                    y_pred=self.predicted_labels)

        print(metrics)

    def calculate_metrics_multiclass(self, y_true, y_pred):
        """
        Calculate Precision, Recall, and F1-Score for multi-class classification.

        Parameters:
            y_true (list): List of actual class labels.
            y_pred (list): List of predicted class labels.

        Returns:
            dict: Dictionary containing Precision, Recall, and F1-Score for each class and their averages.
        """
        metrics = {}
        total_precision, total_recall, total_f1 = 0, 0, 0

        for c in range(self.num_of_classes):
            # Calculate TP, FP, FN for class c
            TP = sum((yt == c and yp == c) for yt, yp in zip(y_true, y_pred))
            FP = sum((yt != c and yp == c) for yt, yp in zip(y_true, y_pred))
            FN = sum((yt == c and yp != c) for yt, yp in zip(y_true, y_pred))

            # Precision, Recall, F1-Score for class c
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Store metrics for this class
            metrics[f"Class {c}"] = {
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1_score
            }

            # Aggregate for averaging
            total_precision += precision
            total_recall += recall
            total_f1 += f1_score

        # Calculate macro-averaged metrics
        avg_precision = total_precision / self.num_of_classes
        avg_recall = total_recall / self.num_of_classes
        avg_f1_score = total_f1 / self.num_of_classes

        metrics["Average"] = {
            "Precision": avg_precision,
            "Recall": avg_recall,
            "F1-Score": avg_f1_score
        }

        return metrics
