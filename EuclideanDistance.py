import math
from pandas import Series
import numpy as np


def calculate_distance_euclidean_numpy(A: np.ndarray, B: np.ndarray) -> float:
    # Ensure both arrays are the same length by padding with zeros
    max_len = max(len(A), len(B))
    A = np.pad(A, (0, max_len - len(A)), constant_values=0)
    B = np.pad(B, (0, max_len - len(B)), constant_values=0)

    # Calculate the Euclidean distance
    return np.sqrt(np.sum((A - B) ** 2))


def calculate_distance_euclidean(A: Series, B: Series) -> float:
    max_len = max(len(A), len(B))
    summation = 0
    for i in range(max_len):
        if i >= len(A):
            summation += math.pow(B.iloc[i], 2)
        elif i >= len(B):
            summation += math.pow(A.iloc[i], 2)
        else:
            summation += math.pow((A.iloc[i] - B.iloc[i]), 2)
    return math.sqrt(summation)
