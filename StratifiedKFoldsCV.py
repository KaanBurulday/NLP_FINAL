import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from knn import KNN

class StratifiedKFoldsCV:
    def __init__(self, **kwargs):
        self.max_workers = kwargs.pop('max_workers', cpu_count())
        self.K = kwargs.get('K', 10)
        self.data_base_path = kwargs.get('data_base_path', None)
        if self.data_base_path is None:
            raise ValueError('data_base_path needs to be provided')

        self.folds_created = kwargs.get('folds_created', False)
        self.folds = kwargs.get('folds')
        self.data = kwargs.get('data')
        self.classes = self.data['class'].unique()
        self.num_of_classes = len(self.classes)

        self.knn_config = kwargs.get('knn_config', None)
        if self.knn_config is None:
            raise ValueError('knn_config needs to be provided')

        # Precompute NumPy arrays for all folds
        self.folds_numpy = {fold: self.folds[fold].reset_index().to_numpy() for fold in self.folds}

        # Precompute combined training data for each fold
        self.combined_train_data = {
            fold: pd.concat(
                [self.folds[train_fold] for train_fold in self.folds if train_fold != fold]
            ).reset_index().to_numpy()
            for fold in self.folds
        }

        self.predicted_labels = []
        self.actual_labels = []

    def process_fold(self, fold):
        """Process a single fold for KNN."""
        start_time = time.time()
        knn_config = self.knn_config.copy()
        knn_config['train_data'] = self.combined_train_data[fold]
        knn = KNN(**knn_config)
        result = knn.predict_bulk(self.folds_numpy[fold])
        duration = time.time() - start_time
        print(f"Fold {fold} is completed. Duration: {duration:.2f}s")
        return {'predictions': result['predictions'], 'actual': result['actual']}

    def start_knn(self):
        """Start KNN using multiprocessing."""
        print(f"Starting KNN with {self.max_workers} processes")
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_fold, fold): fold for fold in self.folds}

            for future in as_completed(futures):
                fold = futures[future]
                try:
                    result = future.result()
                    self.predicted_labels += result['predictions']
                    self.actual_labels += result['actual']
                except Exception as e:
                    print(f"An error occurred with fold {fold}: {e}")

        metrics = self.calculate_metrics_multiclass(y_true=self.actual_labels,
                                                    y_pred=self.predicted_labels)
        return metrics

    def calculate_metrics_multiclass(self, y_true, y_pred):
        """Calculate metrics for multi-class classification."""
        metrics = {}
        total_precision, total_recall, total_f1 = 0, 0, 0

        for c in self.classes:
            TP = sum((yt == c and yp == c) for yt, yp in zip(y_true, y_pred))
            FP = sum((yt != c and yp == c) for yt, yp in zip(y_true, y_pred))
            FN = sum((yt == c and yp != c) for yt, yp in zip(y_true, y_pred))

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics[f"{c}"] = {"Precision": precision, "Recall": recall, "F1-Score": f1_score}
            total_precision += precision
            total_recall += recall
            total_f1 += f1_score

        metrics["Average"] = {
            "Precision": total_precision / self.num_of_classes,
            "Recall": total_recall / self.num_of_classes,
            "F1-Score": total_f1 / self.num_of_classes,
        }
        return metrics
