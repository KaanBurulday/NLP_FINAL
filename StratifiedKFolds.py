import random
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from pandas import DataFrame
from joblib import Parallel, delayed

import knn
from concurrent.futures import ThreadPoolExecutor, as_completed


class StratifiedKFolds:
    def __init__(self, **kwargs):
        self.data: DataFrame = kwargs.get("data", DataFrame())
        self.k_folds: int = kwargs.get("k_folds", 10)
        self.worker_count: int = kwargs.get("worker_count", 4)
        self.precalculate_distances: bool = kwargs.get("precalculate_distances", True)
        self.use_numpy = kwargs.get("use_numpy", True)

    def get_class_percentages(self):
        """Assuming the last column of the row is denoting the class of the row"""
        class_counter = Counter()
        for _, row in self.data.iterrows():
            class_counter[row['class']] += 1
        class_percentages = {}
        for class_ in class_counter.keys():
            class_percentages[class_] = class_counter[class_] / len(self.data)
        return class_percentages

    def get_document_counts(self):
        class_percentages = self.get_class_percentages()
        fold_size = len(self.data) / self.k_folds
        document_count_for_classes = {}
        for class_ in class_percentages.keys():
            document_count_for_classes[class_] = class_percentages[class_] * fold_size
        rounded_document_count_for_classes = {}
        for class_ in document_count_for_classes.keys():
            rounded_document_count_for_classes[class_] = round(document_count_for_classes[class_])
        rounded_total = sum(rounded_document_count_for_classes.values())
        class_count_fractional_remainder = {}
        if rounded_total > self.k_folds:
            for class_ in document_count_for_classes.keys():
                class_count_fractional_remainder[class_] = rounded_document_count_for_classes[class_] - \
                                                           document_count_for_classes[class_]
            max_remainder_class = max(class_count_fractional_remainder, key=class_count_fractional_remainder.get)
            rounded_document_count_for_classes[max_remainder_class] = document_count_for_classes[
                max_remainder_class].__floor__()
        elif rounded_total < self.k_folds:
            for class_ in document_count_for_classes.keys():
                class_count_fractional_remainder[class_] = rounded_document_count_for_classes[class_] - \
                                                           document_count_for_classes[class_]
            max_remainder_class = min(class_count_fractional_remainder, key=class_count_fractional_remainder.get)
            rounded_document_count_for_classes[max_remainder_class] = document_count_for_classes[
                max_remainder_class].__ceil__()
        return rounded_document_count_for_classes

    def split_by_classes(self):
        split_data_by_classes = {}
        for class_ in self.data.iloc[:, -1].unique():
            split_data_by_classes[class_] = []
            for _, row in self.data.loc[self.data.iloc[:, -1] == class_].iterrows():
                split_data_by_classes[class_].append(row)
        return split_data_by_classes

    def create_folds(self):
        # 1. split the data by classes into a dictionary as class : [seriesA, seriesB]
        split_data_by_classes = self.split_by_classes()

        document_count_for_classes = self.get_document_counts()
        remaining_docs = {class_: [] for class_ in document_count_for_classes.keys()}
        folds = {}
        for fold in range(self.k_folds):
            folds[f"fold{fold}"] = {}
            for class_ in document_count_for_classes.keys():
                population_size = len(split_data_by_classes[class_])
                required_count = document_count_for_classes[class_]

                # Rounding up the document count may cause overshoot
                if required_count > population_size:
                    print(
                        f"Warning: Class {class_} has fewer samples ({population_size}) than required ({required_count})")
                    required_count = population_size

                random_indices = random.sample(range(population_size), required_count)
                random_indices.sort(reverse=True)
                random_documents = [split_data_by_classes[class_].pop(i) for i in random_indices]
                folds[f"fold{fold}"][class_] = random_documents

                if fold == self.k_folds - 1:
                    remaining_docs[class_].extend(split_data_by_classes[class_])

        for class_, leftovers in remaining_docs.items():
            folds[f"fold{len(folds) - 1}"][class_].extend(leftovers)

        return folds

    def calculate_cosine_similarities_optimized(self) -> dict:
        doc_ids = self.data.iloc[:, 0].to_numpy()  # First column
        feature_data = self.data.iloc[:, 1:-1].to_numpy()  # Ignore first and last columns

        magnitudes = np.linalg.norm(feature_data, axis=1)

        similarities = {}

        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):  # Avoid duplicate computations and self-comparisons
                if magnitudes[i] == 0 or magnitudes[j] == 0:  # Handle zero vectors
                    sim = 0.0
                else:
                    sim = np.dot(feature_data[i], feature_data[j]) / (magnitudes[i] * magnitudes[j])
                similarities[f"{min(doc_ids[i], doc_ids[j])}-{max(doc_ids[i], doc_ids[j])}"] = sim

        return similarities

    def compute_similarity(self, i, j, feature_data, magnitudes, doc_ids):
        if magnitudes[i] == 0 or magnitudes[j] == 0:
            sim = 0.0
        else:
            sim = np.dot(feature_data[i], feature_data[j]) / (magnitudes[i] * magnitudes[j])
        return f"{min(doc_ids[i], doc_ids[j])}-{max(doc_ids[i], doc_ids[j])}", sim

    def calculate_cosine_similarities_parallel(self) -> dict:
        doc_ids = self.data.iloc[:, 0].to_numpy()
        feature_data = self.data.iloc[:, 1:-1].to_numpy()
        magnitudes = np.linalg.norm(feature_data, axis=1)

        results = Parallel(n_jobs=self.worker_count)(
            delayed(self.compute_similarity)(i, j, feature_data, magnitudes, doc_ids)
            for i in range(len(doc_ids)) for j in range(i + 1, len(doc_ids))
        )

        return dict(results)

    def compute_euclidean(self, i, j, feature_data, doc_ids):
        # Compute the Euclidean distance between two vectors
        distance = np.linalg.norm(feature_data[i] - feature_data[j])
        return f"{min(doc_ids[i], doc_ids[j])}-{max(doc_ids[i], doc_ids[j])}", distance

    def calculate_euclidean_distances_parallel(self) -> dict:
        # Extract document IDs and feature data
        doc_ids = self.data.iloc[:, 0].to_numpy()
        feature_data = self.data.iloc[:, 1:-1].to_numpy()

        # Compute pairwise Euclidean distances in parallel
        results = Parallel(n_jobs=self.worker_count)(
            delayed(self.compute_euclidean)(i, j, feature_data, doc_ids)
            for i in range(len(doc_ids)) for j in range(i + 1, len(doc_ids))
        )

        return dict(results)

    def process_fold(self, fold_idx, folds, knn_k, classifier, precomputed_distances):
        # Separate validation and training folds
        validation_fold = folds[f"fold{fold_idx}"]
        training_folds = {f: folds[f] for f in folds if f != f"fold{fold_idx}"}

        # Flatten validation and training folds into DataFrames
        validation_data_df = pd.DataFrame([doc for cls_docs in validation_fold.values() for doc in cls_docs])
        training_data_df = pd.DataFrame(
            [doc for fold_docs in training_folds.values() for cls_docs in fold_docs.values() for doc in cls_docs])

        # Initialize metrics storage for each class
        class_metrics = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})

        # Iterate through validation samples
        for _, test_sample in validation_data_df.iterrows():
            predicted_class = classifier(knn_k, training_data_df, test_sample[:-1],
                                                use_numpy=self.use_numpy, precomputed_distances=precomputed_distances)
            true_class = test_sample['class']

            if predicted_class == true_class:
                class_metrics[true_class]['TP'] += 1
            else:
                class_metrics[predicted_class]['FP'] += 1
                class_metrics[true_class]['FN'] += 1

        # Calculate precision, recall, and F1-score per class
        results_per_class = {}
        for class_label, metrics in class_metrics.items():
            TP = metrics['TP']
            FP = metrics['FP']
            FN = metrics['FN']
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results_per_class[class_label] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'TP': TP,
                'FP': FP,
                'FN': FN
            }

        return {
            'fold': fold_idx + 1,
            'results_per_class': results_per_class
        }

    def stratified_cross_validation_parallel(self, folds, k, knn_function, precomputed_distances=None):
        # Perform stratified k-fold cross-validation
        results = []

        with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
            future_to_fold = {
                executor.submit(self.process_fold, fold_idx, folds, k, knn_function, precomputed_distances): fold_idx
                for fold_idx in range(len(folds))
            }

            for future in as_completed(future_to_fold):
                fold_idx = future_to_fold[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    print(f"Fold {fold_idx} generated an exception: {exc}")

        # Last minute changes
        # results = []
        #
        # if self.precalculate_distances:
        #     start_time = time.time()
        #     precalculated_distances = self.calculate_cosine_similarities_parallel() if knn_function == knn.classify_with_cs else self.calculate_euclidean_distances_parallel()
        #     print(f"Pre-Calculation of Distance Duration: {time.time() - start_time} seconds")
        # else:
        #     precalculated_distances = None
        #
        # with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
        #     future_to_fold = {
        #         executor.submit(self.process_fold, fold_idx, folds, k, knn_function, precalculated_distances): fold_idx
        #         for fold_idx in range(len(folds))
        #     }
        #
        #     for future in as_completed(future_to_fold):
        #         fold_idx = future_to_fold[future]
        #         try:
        #             result = future.result()
        #             results.append(result)
        #         except Exception as exc:
        #             print(f"Fold {fold_idx} generated an exception: {exc}")

        # Combine results across all folds
        overall_metrics = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
        for fold_result in results:
            for class_label, metrics in fold_result['results_per_class'].items():
                overall_metrics[class_label]['TP'] += metrics['TP']
                overall_metrics[class_label]['FP'] += metrics['FP']
                overall_metrics[class_label]['FN'] += metrics['FN']

        # Calculate overall precision, recall, F1-score, and averages
        class_results = {}
        all_TP, all_FP, all_FN = 0, 0, 0
        for class_label, metrics in overall_metrics.items():
            TP = metrics['TP']
            FP = metrics['FP']
            FN = metrics['FN']
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            class_results[class_label] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'TP': TP,
                'FP': FP,
                'FN': FN
            }

            all_TP += TP
            all_FP += FP
            all_FN += FN

        # Macro Average
        macro_avg = {
            'precision': sum(r['precision'] for r in class_results.values()) / len(class_results),
            'recall': sum(r['recall'] for r in class_results.values()) / len(class_results),
            'f1_score': sum(r['f1_score'] for r in class_results.values()) / len(class_results),
        }

        # Micro Average
        micro_precision = all_TP / (all_TP + all_FP) if (all_TP + all_FP) > 0 else 0
        micro_recall = all_TP / (all_TP + all_FN) if (all_TP + all_FN) > 0 else 0
        micro_f1_score = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (
                                                                                                          micro_precision + micro_recall) > 0 else 0

        micro_avg = {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1_score': micro_f1_score,
        }

        return {
            'class_results': class_results,
            'macro_avg': macro_avg,
            'micro_avg': micro_avg
        }
