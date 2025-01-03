import heapq
import random
import time
from collections import Counter

import numpy as np
from pandas import DataFrame, Series
from CosineSimilarity import distance_cs, distance_cs_numpy
from EuclideanDistance import calculate_distance_euclidean


def classify_with_cs(n: int, data: DataFrame, test_sample: Series, use_numpy: bool, precomputed_distances: dict = None):
    """
    Assuming rows (series) of the dataframe is like below:
        key    termA termB termC ... class
        Doc100 0.0   0.234 0.003 ...  1
    And the data of the test sample should be like below:
        key     termA termB termC ...
        newDoc  0.003 0.013 0.023 ...
    """
    start_time = time.time()
    distances = []  # docName, distance, class
    doc_ids = data['Doc'].to_numpy()
    classes = data['class'].to_numpy()
    if precomputed_distances is not None:
        for i in range(len(data)):
            cs_key = f"{min(doc_ids[i], test_sample['Doc'])}-{max(doc_ids[i], test_sample['Doc'])}"
            distance = precomputed_distances.get(cs_key)
            if distance is None:
                raise KeyError(f"Cosine similarity key not found for {doc_ids[i]} and {test_sample['Doc']}")

            distances.append((doc_ids[i], distance, classes[i]))
    else:
        if use_numpy:
            data_array = data.to_numpy()
            for i in range(len(data)):
                doc_vector = data_array[i, 1:-1]
                test_vector = test_sample[1:].to_numpy()
                distance = distance_cs_numpy(doc_vector, test_vector)  # numpy provides much better performance
                distances.append((doc_ids[i], distance, classes[i]))
        else:
            for i in range(len(data)):
                distance = distance_cs(data.iloc[i][1:-1], test_sample[1:])
                distances.append((doc_ids[i], distance, classes[i]))

    # Get the top N neighbors
    top_n = heapq.nlargest(n, distances, key=lambda x: x[1])

    # Count class votes
    class_votes = Counter([neighbor[2] for neighbor in top_n])
    max_votes = max(class_votes.values())
    top_classes = [cls for cls, count in class_votes.items() if count == max_votes]
    choice = random.choice(top_classes)
    # print(f"Predicted class: {choice} Time: {time.time() - start_time}s")
    # Handle ties by selecting randomly among top classes
    return choice


def classify_with_euclidean(
        n: int, data: DataFrame, test_sample: Series, use_numpy: bool, precomputed_distances: dict = None
):
    """
    Classify a test sample using k-NN with Euclidean distance.

    :param n: Number of nearest neighbors to consider.
    :param data: DataFrame where each row represents a document with features and a class.
    :param test_sample: Series representing the test sample to classify.
    :param use_numpy: Whether to use NumPy for distance calculation.
    :param precomputed_distances: Dictionary of precomputed Euclidean distances.
    :return: Predicted class for the test sample.
    """
    distances = []  # Stores tuples of (doc_id, distance, class)
    doc_ids = data["Doc"].to_numpy()
    classes = data["class"].to_numpy()

    if precomputed_distances is not None:
        for i in range(len(data)):
            dist_key = f"{min(doc_ids[i], test_sample['Doc'])}-{max(doc_ids[i], test_sample['Doc'])}"
            distance = precomputed_distances.get(dist_key)
            if distance is None:
                raise KeyError(f"Euclidean distance key not found for {doc_ids[i]} and {test_sample['Doc']}")

            distances.append((doc_ids[i], distance, classes[i]))
    else:
        # Dynamically compute distances
        if use_numpy:
            data_array = data.iloc[:, 1:-1].to_numpy()
            test_vector = test_sample[1:].to_numpy()
            for i in range(len(data)):
                distance = np.linalg.norm(data_array[i] - test_vector)
                distances.append((doc_ids[i], distance, classes[i]))
        else:
            for i in range(len(data)):
                distance = calculate_distance_euclidean(data.iloc[i][1:-1], test_sample[1:])
                distances.append((doc_ids[i], distance, classes[i]))

    # Get the top N nearest neighbors (smallest distances)
    top_n = heapq.nsmallest(n, distances, key=lambda x: x[1])  # Smallest distances for Euclidean

    # Count class votes among top N neighbors
    class_votes = Counter([neighbor[2] for neighbor in top_n])
    max_votes = max(class_votes.values())
    top_classes = [cls for cls, count in class_votes.items() if count == max_votes]
    choice = random.choice(top_classes)  # Handle ties randomly

    return choice
