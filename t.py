import cProfile
import pathlib
import pstats
from collections import defaultdict

import pandas as pd


create_tf_idf_table_to_path = f"{pathlib.Path().resolve()}\\TF_IDF_V2_Table.parquet"

df = pd.read_parquet(create_tf_idf_table_to_path)

print(df.shape[0])

print(df)

np_df = df.reset_index().to_numpy()

print(np_df)

#tf_idf_table = pd.read_parquet(create_tf_idf_table_to_path).reset_index().to_numpy()

print("--------------------------------------------")

#print(tf_idf_table[int(np_df[0][0])][1:-1])

# print(tf_idf_table[np_df[0][0]][1:-1])

train_path = f"{pathlib.Path().resolve()}\\folds\\0\\train0.csv"
test_path = f"{pathlib.Path().resolve()}\\folds\\0\\test0.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

print(test_data)

print(test_data.columns)

train_data_np = train_data.to_numpy()
test_data_np = test_data.to_numpy()

#print(test_data_np)

tf_idf_table = pd.read_parquet(create_tf_idf_table_to_path).reset_index().to_numpy()

#print(tf_idf_table)

from knn import KNN

knn_config = {
    'k': 10,
    'train_data': train_data_np,
    'tf_idf': tf_idf_table,
}

def test():
    # distances = DistanceCalculator.calculate_distances_between_test_and_train_mt(train_data_np, test_data_np, tf_idf_table)
    # print(distances)
    knn = KNN(**knn_config)
    correct_prediction = 0
    prediction_count = 0
    for row in test_data_np:
        prediction_count += 1
        prediction = knn.predict(query=row)
        print(f"Prediction: {prediction}")
        print(f"Actual: {row[-1]}")
        if row[-1] == prediction:
            correct_prediction += 1
        print(f"{correct_prediction}/{prediction_count}")

def test1():
    # distances = DistanceCalculator.calculate_distances_between_test_and_train_mt(train_data_np, test_data_np, tf_idf_table)
    # print(distances)
    knn = KNN(**knn_config)
    class_metrics = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    prediction_count = 0
    for row in test_data_np:
        prediction = knn.predict(query=row)
        prediction_count += 1
        true_class = row[-1]

        if prediction == true_class:
            class_metrics[true_class]['TP'] += 1
        else:
            class_metrics[prediction]['FP'] += 1
            class_metrics[true_class]['FN'] += 1
        print(f"{prediction_count}/{test_data_np.shape[0]} prediction completed.")

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
            'results_per_class': results_per_class
        }

def stratified_cross_validation():
    # Perform stratified k-fold cross-validation
    results = []

    result = test1()
    results.append(result)

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

def create_results_table():
    results = stratified_cross_validation()

    class_results = results['class_results']
    macro_avg = results['macro_avg']
    micro_avg = results['micro_avg']

    table = {
        'Class': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': [],
        'True Positives': [],
        'False Positives': [],
        'False Negatives': [],
    }

    for class_label, metrics in class_results.items():
        table['Class'].append(class_label)
        table['Precision'].append(metrics['precision'])
        table['Recall'].append(metrics['recall'])
        table['F1-Score'].append(metrics['f1_score'])
        table['True Positives'].append(metrics['TP'])
        table['False Positives'].append(metrics['FP'])
        table['False Negatives'].append(metrics['FN'])

    # Add Macro and Micro Averages
    table['Class'].extend(['Macro Avg', 'Micro Avg'])
    table['Precision'].extend([macro_avg['precision'], micro_avg['precision']])
    table['Recall'].extend([macro_avg['recall'], micro_avg['recall']])
    table['F1-Score'].extend([macro_avg['f1_score'], micro_avg['f1_score']])
    table['True Positives'].extend(['-', '-'])
    table['False Positives'].extend(['-', '-'])
    table['False Negatives'].extend(['-', '-'])


    pd_table = pd.DataFrame(table)
    pd_table.to_csv("results_table.csv", index=False)
    print(pd_table)
    return pd.DataFrame(table)

cProfile.run('test()', 'profile_output')
p = pstats.Stats('profile_output')
p.sort_stats(pstats.SortKey.TIME).print_stats(10)

