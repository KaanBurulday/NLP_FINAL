import pathlib
import time

import pandas as pd
from tqdm import tqdm

import TF_IDF
import knn
import StratifiedKFolds


##################### TF-IDF #####################
start_time = time.time()
assignment_data_path = f"{pathlib.Path().resolve()}\\makaleler-yazarlar"
create_tf_idf_table_to_path = f"{pathlib.Path().resolve()}\\TF_IDF_Table.csv"
create_tf_idf_table = True # If you want to recreate the tf idf table, make it True

use_nltk = False # To use BPE make this False
use_files = True
file_recreate = True

turkish_stopwords = [
    "acaba", "ama", "ancak", "aslında", "az", "bazı", "belki", "biri", "birkaç", "birşey",
    "biz", "bu", "çok", "çünkü", "da", "daha", "de", "defa", "diye", "eğer", "en", "gibi",
    "hem", "hep", "hepsi", "her", "hiç", "ile", "ise", "kez", "ki", "kim", "mı", "mu",
    "mü", "nasıl", "ne", "neden", "nerde", "nerede", "nereye", "niçin", "niye", "o",
    "sanki", "şey", "siz", "şu", "tüm", "ve", "veya", "ya", "yani"
]

TF_IDF_config = {
    "data_path": assignment_data_path,
    "create_tf_idf_table_to_path": create_tf_idf_table_to_path,
    "corpus_path": f"{pathlib.Path().resolve()}\\corpus.txt",
    "vocab_path": f"{pathlib.Path().resolve()}\\vocab.txt",
    "make_corpus_unique": True,
    "stop_words": turkish_stopwords,
    "word_split_regex": ' ',
    "only_alpha": True,
    "use_nltk_stemmer": False,
    # for BPE algorithm, if you choose nltk, the below parameters are not important
    "end_of_word_token": "_",
    "n": 10000,
    "show_bpe_counter": True,
}

use_numpy = True

# with nltk between 40-60 seconds, with BPE between 40-100 seconds. Probably due to use of and iterating dataframes/series
if create_tf_idf_table:
    tf_idf = TF_IDF.TF_IDF(**TF_IDF_config)
    tf_idf.create_tf_idf_table((use_files, file_recreate), use_nltk)

tf_idf_table = pd.read_csv(create_tf_idf_table_to_path)
print(tf_idf_table)

print(f"\nTF-IDF Table Creation Duration:{time.time() - start_time}s")
##################### TF-IDF #####################

def create_results_table(results):
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

    return pd.DataFrame(table)

##################### Stratified K-Folds Cross Validation #####################
# Takes about 80-100 seconds on r5 7600 cpu (6 cores 12 threads)
start_time = time.time()
knn_function = knn.classify_with_cs
knn_k = 4
k_folds = 10

StratifiedKFolds_config = {
    "data": tf_idf_table,
    "k_folds": k_folds,
    "worker_count": 6,
    "precalculate_cosine_similarity": True,
    "use_numpy": use_numpy
}

SK_Folder = StratifiedKFolds.StratifiedKFolds(**StratifiedKFolds_config)
folds = SK_Folder.create_folds()

print(f"Starting Pre-Calculation Phase.")
pre_calculation_start_time = time.time()
precalculated_cosine_distances = SK_Folder.calculate_cosine_similarities_optimized()
precalculated_euclidean_distances = SK_Folder.calculate_cosine_similarities_optimized()
duration_pre_calculation = time.time() - pre_calculation_start_time
print(f"Pre-Calculation Duration: {duration_pre_calculation:.2f}s")

for i in tqdm(range(knn_k), desc="Evaluating k-NN for different k values"):
    k = i + 1

    # Cosine Similarity
    start_time = time.time()
    sk_results_cs = SK_Folder.stratified_cross_validation_parallel(
        folds=folds, k=k, knn_function=knn.classify_with_cs, precomputed_distances=precalculated_cosine_distances
    )
    duration_cs = time.time() - start_time
    print(f"Cosine Similarity | K: {k}, Duration: {duration_cs:.2f}s")

    results_table_cs = create_results_table(sk_results_cs)
    if use_nltk:
        results_table_cs.to_csv(f"Result_Table_NLTK_Cs_K_{k}.csv", index=False)
    else:
        results_table_cs.to_csv(f"Result_Table_BPE_CS_K_{k}.csv", index=False)

    # Euclidean Distance
    start_time = time.time()
    sk_results_euc = SK_Folder.stratified_cross_validation_parallel(
        folds=folds, k=k, knn_function=knn.classify_with_euclidean, precomputed_distances=precalculated_euclidean_distances
    )
    duration_euc = time.time() - start_time
    print(f"Euclidean Distance | K: {k}, Duration: {duration_euc:.2f}s")

    results_table_euc = create_results_table(sk_results_euc)
    if use_nltk:
        results_table_euc.to_csv(f"Result_Table_NLTK_Euc_K_{k}.csv", index=False)
    else:
        results_table_euc.to_csv(f"Result_Table_BPE_Euc_K_{k}.csv", index=False)

# for i in range(knn_k):
#     sk_results = SK_Folder.stratified_cross_validation_parallel(folds=folds, k=i+1, knn_function = knn.classify_with_cs)
#
#     print(
#         f"Stratified {k_folds}-Folds Cross Validation For {i+1}-NN Classification Duration: {time.time() - start_time} seconds")
#
#     results_table = create_results_table(sk_results)
#     print(f"K for KNN: {i+1}, Token Type: {'NLTK' if use_nltk else 'BPE'}, Method: Cosine Similarity")
#     print(results_table)
#     results_table.to_csv(f"Result_Table_Cs_K_{i+1}.csv", index=False)
#
#
#     sk_results = SK_Folder.stratified_cross_validation_parallel(folds=folds, k=i+1, knn_function = knn.classify_with_euclidean)
#
#     print(
#         f"Stratified {k_folds}-Folds Cross Validation For {i+1}-NN Classification Duration: {time.time() - start_time} seconds")
#
#     results_table = create_results_table(sk_results)
#     print(f"K for KNN: {i+1}, Token Type: {'NLTK' if use_nltk else 'BPE'}, Method: Euclidean")
#     print(results_table)
#     results_table.to_csv(f"Result_Table_Euc_K_{i+1}.csv", index=False)