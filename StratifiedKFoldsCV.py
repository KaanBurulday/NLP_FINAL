# Mission:
# 1. Create a folds folder
# 2.
# 3.
#
from collections import Counter

from pandas import DataFrame

from TextSanitizer import sanitizer_string
import os


class StratifiedKFoldsCV:
    def __init__(self, **kwargs):
        self.K = kwargs.get('K', 10)
        self.data_base_path = kwargs.get('data_base_path', None)
        if self.data_base_path is not None:
            self.stop_words = kwargs.get('stop_words', None)
            self.only_alpha = kwargs.get('only_alpha', True)
            self.split_regex = kwargs.get('split_regex', ' ')
            self.data = self.get_data()
        else:
            self.data = kwargs.get('data', None)

        self.class_counter = self.get_class_counts()
        self.class_percentages = self.get_class_percentages()
        self.data_count_per_fold = self.get_data_count_per_fold()
        self.class_data_count_per_fold = self.get_data_count_per_folds()


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

    def get_data(self):
        data = []
        base_path = self.data_base_path
        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    with open(file_path, 'r') as f:
                        text = sanitizer_string(text=f.read(), stop_words=self.stop_words, only_alpha=self.only_alpha,
                                                split_regex=self.split_regex)
                        data.append({'text': text, 'class': folder})
        return DataFrame(data)

    def get_data_count_per_fold(self):
        # Determine the amount of data will be in each fold
        return len(self.data) / self.K

    def get_data_count_per_folds(self):
        # the amount of data will be for each class in each fold
        class_data_count_per_fold = {}
        for key in self.class_counter.keys():
            class_data_count_per_fold[key] = (self.data_count_per_fold*self.class_percentages[key])/100
        return class_data_count_per_fold


    def create_folds(self):
        pass
