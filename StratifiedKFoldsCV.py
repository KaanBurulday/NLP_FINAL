
import pathlib
from collections import Counter
from pandas import DataFrame
from TextSanitizer import sanitizer_string
import os


class StratifiedKFoldsCV:
    def __init__(self, **kwargs):
        self.K = kwargs.get('K', 10)


    def create_folds(self):
        pass





