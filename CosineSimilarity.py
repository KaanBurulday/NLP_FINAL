import math
import numpy as np
from pandas import Series

def precomputed_distance_cs_numpy(A: np.ndarray, B: np.ndarray, magA: float, magB: float) -> float:
    if magA == 0 or magB == 0:
        return 0
    return np.dot(A, B) / (magA * magB)

def distance_cs_numpy(A: np.ndarray, B: np.ndarray) -> float:
    if np.linalg.norm(A) == 0 or np.linalg.norm(B) == 0:
        return 0
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def magnitude_numpy(A: np.ndarray) -> float:
    return float(np.linalg.norm(A))


def distance_cs(A: Series, B: Series) -> float:
    if magnitude(A) == 0 or magnitude(B) == 0:
        return 0
    return dot_product(A, B) / (magnitude(A) * magnitude(B))


def dot_product(A: Series, B: Series) -> float:
    min_len = min(len(A), len(B))
    summation = 0
    for i in range(min_len):
        summation += A.iloc[i] * B.iloc[i]
    return summation


def magnitude(series: Series) -> float:
    summation = 0
    for i in range(len(series)):
        summation += math.pow(series.iloc[i], 2)
    return math.sqrt(summation)
