import json
import pathlib
from collections import Counter
from sklearn.model_selection import StratifiedKFold
import pandas
from pandas import DataFrame
import os
import pickle


class FoldCreator:
    # given a data, it will create K folds.

    def __init__(self, **kwargs):
        self.K = kwargs.get('K', 10)
        self.data = kwargs.get('data', None)
        if self.data is None:
            raise ValueError("Data must not be None!")

        self.folds = {}

        self.use_sklearn = kwargs.get('use_sklearn', True)
        if self.use_sklearn:
            self.skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            X = self.data['text']
            y = self.data['class']
            for i, (train_index, test_index) in enumerate(self.skf.split(X, y)):
                self.folds[i] = self.data.iloc[test_index]
        else:
            self.class_counter = self.get_class_counts()
            self.class_percentages = self.get_class_percentages()
            self.data_count_per_fold = self.get_data_count_per_fold()
            self.class_data_count_per_fold = self.get_class_data_count_per_folds()

            self.create_folds()

    def get_class_counts(self):
        class_counter = Counter()
        for index, row in self.data.iterrows():
            class_counter[row['class']] += 1
        return class_counter

    def get_class_percentages(self):
        class_percentages = {}
        for key in self.class_counter.keys():
            class_percentages[key] = round((self.class_counter[key] * 100) / len(self.data), 4)
        return class_percentages

    # def get_data(self):
    #     data = []
    #     base_path = self.data_base_path
    #     for folder in os.listdir(base_path):
    #         folder_path = os.path.join(base_path, folder)
    #         if os.path.isdir(folder_path):
    #             for file in os.listdir(folder_path):
    #                 file_path = os.path.join(folder_path, file)
    #                 with open(file_path, 'r') as f:
    #                     text = sanitizer_string(text=f.read(), stop_words=self.stop_words, only_alpha=self.only_alpha,
    #                                             split_regex=self.split_regex)
    #                     data.append({'text': text, 'class': folder})
    #     return DataFrame(data)

    def get_data_count_per_fold(self):
        # Determine the amount of data will be in each fold
        return len(self.data) / self.K

    def get_class_data_count_per_folds(self):
        # the amount of data will be for each class in each fold
        class_data_count_per_fold = {}
        for key in self.class_counter.keys():
            class_data_count_per_fold[key] = int(round((self.data_count_per_fold * self.class_percentages[key]) / 100))
        return class_data_count_per_fold

    def create_folds(self):
        data_copy = self.data.copy().sample(frac=1).reset_index(drop=True)
        for i in range(self.K):
            self.folds[i] = DataFrame()
            for key in self.class_data_count_per_fold.keys():
                filtered_data = data_copy[data_copy['class'] == key]
                desired_n = self.class_data_count_per_fold[key]
                available_n = len(filtered_data)
                if desired_n > available_n:
                    print(
                        f"Warning: Not enough rows to sample from class '{key}'! "
                        f"Requested {desired_n}, only {available_n} available."
                    )
                    desired_n = available_n
                sub_data_sample = filtered_data.sample(n=desired_n)
                data_copy.drop(sub_data_sample.index, inplace=True)
                self.folds[i] = pandas.concat([self.folds[i], sub_data_sample], axis=0)

        if not data_copy.empty:
            print("Warning: Some data might not be allocated to folds or there's leftover data. The leftover data will "
                  "be distributed to each fold starting from fold 1")
            i = 0
            for row in data_copy.iterrows():
                self.folds[i] = pandas.concat([self.folds[i], row], axis=0)
                i = (i+1) % self.K


    def save_folds(self):
        # To create folder and files for folds.
        folds_root_path = pathlib.Path().resolve() / "folds"
        os.makedirs(folds_root_path, exist_ok=True)
        for fold in self.folds:
            other_folds = {other_fold for other_fold in self.folds if other_fold != fold}
            train_fold = DataFrame()
            for other_fold in other_folds:
                train_fold = pandas.concat([train_fold, self.folds[other_fold]], axis=0)
            os.makedirs(folds_root_path / f"{fold}", exist_ok=True)

            train_fold.sort_index().to_csv(folds_root_path / f"{fold}" / f"train{fold}.csv", index=True)
            self.folds[fold].sort_index().to_csv(folds_root_path / f"{fold}" / f"test{fold}.csv")

        # with open('folds.pkl', 'wb') as file:
        #     pickle.dump(self.folds, file)

    def save_folds_pkl(self):
        folds_root_path = pathlib.Path().resolve() / "folds.pkl"
        data_to_save = {key: value.to_dict() for key, value in self.folds.items()}
        with open(folds_root_path, 'wb') as file:
            pickle.dump(data_to_save, file)

    def save_folds_json(self):
        folds_root_path = pathlib.Path().resolve() / "folds.json"
        data_to_save = {key: value.to_dict() for key, value in self.folds.items()}
        with open(folds_root_path, 'w') as file:
            json.dump(data_to_save, file)

