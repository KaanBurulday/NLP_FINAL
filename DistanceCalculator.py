from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from CosineSimilarity import precomputed_distance_cs_numpy



def calculate_distances_between_test_and_train(train_data: DataFrame, test_data: DataFrame,
                                               tf_idf_table_path: str):
    """
    :param train:
    :param test:
    :param tf_idf_table_path:
    :return:
    Will return the distances between train and test data.
    """
    train_np = train_data.reset_index().to_numpy()
    test_np = test_data.reset_index().to_numpy()
    tf_idf_table = pd.read_parquet(tf_idf_table_path).reset_index().to_numpy()

    # magnitudes = []
    # for row in tf_idf_table:
    #     magnitudes.append(np.linalg.norm(row[1:-1]))
    # np_magnitudes = np.array(magnitudes)
    # magnitudes = np.zeros(shape=tf_idf_table.shape[0])
    # for row in tf_idf_table:
    #     magnitudes[int(row[0])] = np.linalg.norm(row[1:-1])
    magnitudes = np.linalg.norm(tf_idf_table[:, 1:-1].astype(float), axis=1)

    distances = {}
    for test_row in test_np:
        for train_row in train_np:
            distances[f"{test_row[0]}-{train_row[0]}"] = precomputed_distance_cs_numpy(
                tf_idf_table[int(test_row[0])][1:-1],
                tf_idf_table[int(train_row[0])][1:-1],
                magnitudes[int(test_row[0])],
                magnitudes[int(train_row[0])])
    return distances

def calculate_distances_between_test_and_train_mt(train_data: np.ndarray, test_data: np.ndarray,
                                                  tf_idf_table: np.ndarray):
    """
    :param train_data: Training data DataFrame.
    :param test_data: Test data DataFrame.
    :return: A dictionary containing distances between test and train data.
    """
    # Compute magnitudes for TF-IDF vectors
    # magnitudes = np.zeros(shape=tf_idf_table.shape[0])
    # for row in tf_idf_table:
    #     magnitudes[int(row[0])] = np.linalg.norm(row[1:-1])
    magnitudes = np.linalg.norm(tf_idf_table[:, 1:-1].astype(float), axis=1)

    # Define a function to compute distances for a single test-train pair
    def compute_distance(test_row, train_row):
        test_index, train_index = test_row[0], train_row[0]
        test_vector = tf_idf_table[int(test_index)][1:-1]
        train_vector = tf_idf_table[int(train_index)][1:-1]
        if magnitudes[int(test_index)] == 0 or magnitudes[int(train_index)] == 0:
            distance = 0
        else:
            distance = np.dot(test_vector, train_vector) / (
                    magnitudes[int(test_index)] * magnitudes[int(train_index)])
        return f"{test_index}-{train_index}", round(distance, 6), train_row[-1]

    # Use ThreadPoolExecutor to parallelize distance computations
    distances = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(compute_distance, test_row, train_row)
            for test_row in test_data
            for train_row in train_data
        ]
        for future in as_completed(futures):
            pair, distance, _class = future.result()
            distances[pair] = (distance, _class)

    return distances

def calculate_distances_between_query_and_train(train_data: np.ndarray, query: np.ndarray,
                                                   tf_idf_table: np.ndarray):
    """
    :param train:
    :param test:
    :param tf_idf_table_path:
    :return:
    Will return the distances between train and test data.
    """

    # magnitudes = []
    # for row in tf_idf_table:
    #     magnitudes.append(np.linalg.norm(row[1:-1]))
    # np_magnitudes = np.array(magnitudes)
    # magnitudes = np.zeros(shape=tf_idf_table.shape[0])
    # for row in tf_idf_table:
    #     magnitudes[int(row[0])] = np.linalg.norm(row[1:-1])
    magnitudes = np.linalg.norm(tf_idf_table[:, 1:-1].astype(float), axis=1)

    distances = {}
    for train_row in train_data:
        query_index, train_index = query[0], train_row[0]
        query_vector = tf_idf_table[int(query_index)][1:-1]
        train_vector = tf_idf_table[int(train_index)][1:-1]
        if magnitudes[int(train_index)] == 0 or magnitudes[int(query_index)] == 0:
            distance = 0
        else:
            distance = np.dot(query_vector, train_vector) / (
                    magnitudes[int(query_index)] * magnitudes[int(train_index)])
        distances[f"{query_index}-{train_index}"] = (round(distance, 6), train_row[-1])
    return distances

def calculate_distances_between_query_and_train_mt(train_data: np.ndarray, query: np.ndarray,
                                                   tf_idf_table: np.ndarray):
    """
    :param train_data: Training data DataFrame.
    :param test_data: Test data DataFrame.
    :param tf_idf_table_path: Path to the TF-IDF table file in Parquet format.
    :return: A dictionary containing distances between test and train data.
    """
    # Compute magnitudes for TF-IDF vectors
    # magnitudes = np.zeros(shape=tf_idf_table.shape[0])
    # for row in tf_idf_table:
    #     magnitudes[int(row[0])] = np.linalg.norm(row[1:-1])
    magnitudes = np.linalg.norm(tf_idf_table[:, 1:-1].astype(float), axis=1)

    # Define a function to compute distances for a single test-train pair
    def compute_distance(test_row, train_row):
        test_index, train_index = test_row[0], train_row[0]
        test_vector = tf_idf_table[int(test_index)][1:-1]
        train_vector = tf_idf_table[int(train_index)][1:-1]
        if magnitudes[int(test_index)] == 0 or magnitudes[int(train_index)] == 0:
            distance = 0
        else:
            distance = np.dot(test_vector, train_vector) / (
                    magnitudes[int(test_index)] * magnitudes[int(train_index)])
        return f"{test_index}-{train_index}", round(distance, 6), train_row[-1]

    # Use ThreadPoolExecutor to parallelize distance computations
    distances = {}
    #total_tasks = len(train_data)
    #completed_tasks = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(compute_distance, query, train_row)
            for train_row in train_data
        ]
        for future in as_completed(futures):
            pair, distance, _class = future.result()
            distances[pair] = (distance, _class)
            #completed_tasks += 1
            #if completed_tasks % 100 == 0 or completed_tasks == total_tasks:
                #print(f"Completed {completed_tasks}/{total_tasks} tasks")

    return distances
