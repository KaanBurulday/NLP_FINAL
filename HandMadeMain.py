import cProfile
import math
import os
import pathlib
import pickle
import pstats
import time
import json
import pandas as pd
from pandas import DataFrame

import DistanceCalculator
from TextSanitizer import sanitizer_string
from FoldCreator import FoldCreator
from StratifiedKFoldsCV import StratifiedKFoldsCV
from TF_IDF_V2 import TF_IDF

assignment_data_path = f"{pathlib.Path().resolve()}\\{os.getenv('DATA_FOLDER_NAME')}"
create_tf_idf_table_to_path = f"{pathlib.Path().resolve()}\\TF_IDF_V2_Table.parquet"
results_path = f"{pathlib.Path().resolve()}\\results\\Hand_Made_Results.csv"

folds_path_json = f"{pathlib.Path().resolve()}\\folds.json"
folds_path_pkl = f"{pathlib.Path().resolve()}\\folds.pkl"
recreate_folds = True
recreate_tf_idf = False
if not os.path.isfile(path=create_tf_idf_table_to_path):
    recreate_tf_idf = True

turkish_stopwords = [
    "acaba", "ama", "ancak", "aslında", "az", "bazı", "belki", "biri", "birkaç", "birşey",
    "biz", "bu", "çok", "çünkü", "da", "daha", "de", "defa", "diye", "eğer", "en", "gibi",
    "hem", "hep", "hepsi", "her", "hiç", "ile", "ise", "kez", "ki", "kim", "mı", "mu",
    "mü", "nasıl", "ne", "neden", "nerde", "nerede", "nereye", "niçin", "niye", "o",
    "sanki", "şey", "siz", "şu", "tüm", "ve", "veya", "ya", "yani"
]
TF_IDF_config = {
    "data": None,
    "tf_idf_table_to_path": create_tf_idf_table_to_path,
    "corpus_path": f"{pathlib.Path().resolve()}\\corpus.txt",
    "vocab_path": f"{pathlib.Path().resolve()}\\vocab.txt",
    "make_corpus_unique": True,
    "stop_words": turkish_stopwords,
    "word_split_regex": ' ',
    "only_alpha": True,

    "min_df": 0.001,
    "max_df": 0.55,

    "use_nltk_stemmer": False,
    "use_nltk_tokenizer": True,
    # for BPE algorithm, if you choose nltk, the below parameters are not important
    "end_of_word_token": "_",
    "n": 10000,
    "show_bpe_counter": True,
}
folds_creator_config = {
    'data_base_path': assignment_data_path,
    'K': 10,  # Fold Amount
    'stop_words': turkish_stopwords,
    'only_alpha': True,
    'split_regex': ' ',
    'use_sklearn': True
}
knn_config = {
    'k': 10,  # K closest neighbors
    'train_data': None,
    'method': DistanceCalculator.calculate_distances_between_query_and_train
}
stratified_kfolds_config = {
    'K': 10,  # Fold Amount
    'data_base_path': assignment_data_path,
    'fold_creator_config': folds_creator_config,
    'knn_config': knn_config,
    'max_workers': 5
}


def get_data(data_base_path, stop_words, only_alpha, split_regex):
    data = []
    base_path = pathlib.Path(data_base_path)
    for folder_path in base_path.iterdir():
        if folder_path.is_dir():
            for file_path in folder_path.iterdir():
                if file_path.is_file():
                    with file_path.open('r') as f:
                        text = sanitizer_string(
                            text=f.read(),
                            stop_words=stop_words,
                            only_alpha=only_alpha,
                            split_regex=split_regex
                        )
                        data.append({'text': text, 'class': folder_path.name})
    return DataFrame(data)


def get_folds_json():
    global recreate_folds
    if recreate_folds is False and os.path.isfile(path=folds_path_json):
        folds_df = {}
        with open(folds_path_json, "r") as file:
            folds = json.load(file)
        for key, value in folds.items():
            folds_df[key] = pd.DataFrame(value)

        return folds_df
    else:
        recreate_folds = True

def save_folds_pkl(folds, path):
    data_to_save = {key: value.to_dict() for key, value in folds.items()}
    with open(path, 'wb') as file:
        pickle.dump(data_to_save, file)

def save_folds_json(folds, path):
    data_to_save = {key: value.to_dict() for key, value in folds.items()}
    with open(path, 'w') as file:
        json.dump(data_to_save, file)

def run_knn():
    global fold_creator
    start_time = time.time()
    data = get_data(data_base_path=assignment_data_path,
                    stop_words=turkish_stopwords,
                    only_alpha=True,
                    split_regex=' ')
    knn_config['k'] = int(math.sqrt(len(data)))
    if recreate_folds:
        folds_creator_config['data'] = data
        fold_creator = FoldCreator(**folds_creator_config)
        folds = fold_creator.folds
    else:
        folds = get_folds_json()
    print("--- Folds creation duration: %s seconds ---" % (time.time() - start_time))
    if recreate_tf_idf:
        TF_IDF_config["data"] = data
        tf_idf = TF_IDF(**TF_IDF_config)
        tf_idf.create_tf_idf_table((True, True))
    print("--- TF-IDF table creation duration: %s seconds ---" % (time.time() - start_time))
    tf_idf_table = pd.read_parquet(create_tf_idf_table_to_path).reset_index().to_numpy()
    knn_config["tf_idf_table"] = tf_idf_table
    stratified_kfolds_config['knn_config'] = knn_config
    stratified_kfolds_config['folds_created'] = True
    stratified_kfolds_config['data'] = data
    stratified_kfolds_config['folds'] = folds
    sfkcv = StratifiedKFoldsCV(**stratified_kfolds_config)
    results = sfkcv.start_knn()
    print(
        f"--- Stratified {stratified_kfolds_config['K']}-Folds Cross Validation with {knn_config['k']}-NN classification using Cosine Similarity and TF-IDF Table duration: %s seconds ---" % (
                time.time() - start_time))
    print(results["Average"])
    pd.DataFrame(results).to_csv(results_path)
    if recreate_folds is True:
        save_folds_json(folds=folds, path=folds_path_json)


def bit_of_handmade_run_with_profiler():
    start_time = time.time()
    profile_output_filename = 'profile_outputs\\hand_made_profile_output'
    cProfile.run('run_knn()', profile_output_filename)
    p = pstats.Stats(profile_output_filename)
    p.sort_stats(pstats.SortKey.TIME).print_stats(10)

    print("--- KNN_TF-IDF_CosSim total duration: %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    bit_of_handmade_run_with_profiler()