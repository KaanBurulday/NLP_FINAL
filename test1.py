import cProfile
import pathlib
import pstats
import pandas as pd
from pandas import DataFrame

from TextSanitizer import sanitizer_string
from FoldCreator import FoldCreator
from StratifiedKFoldsCV import StratifiedKFoldsCV
from TF_IDF_V2 import TF_IDF

assignment_data_path = f"{pathlib.Path().resolve()}\\makaleler-yazarlar"
create_tf_idf_table_to_path = f"{pathlib.Path().resolve()}\\TF_IDF_V2_Table.parquet"

turkish_stopwords = [
    "acaba", "ama", "ancak", "aslında", "az", "bazı", "belki", "biri", "birkaç", "birşey",
    "biz", "bu", "çok", "çünkü", "da", "daha", "de", "defa", "diye", "eğer", "en", "gibi",
    "hem", "hep", "hepsi", "her", "hiç", "ile", "ise", "kez", "ki", "kim", "mı", "mu",
    "mü", "nasıl", "ne", "neden", "nerde", "nerede", "nereye", "niçin", "niye", "o",
    "sanki", "şey", "siz", "şu", "tüm", "ve", "veya", "ya", "yani"
]
TF_IDF_config = {
    "data": None,
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

    "min_df": 0.005,
    "max_df": 0.75
}
folds_creator_config = {
    'data_base_path': assignment_data_path,
    'K': 10,
    'stop_words': turkish_stopwords,
    'only_alpha': True,
    'split_regex': ' ',
    'use_sklearn': True
}
knn_config = {
    'k': 10,
    'train_data': None,
}
stratified_kfolds_config = {
    'K': 10,
    'data_base_path': assignment_data_path,
    'fold_creator_config': folds_creator_config,
    'knn_config': knn_config
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


def run_knn():
    data = get_data(data_base_path=assignment_data_path,
                    stop_words=turkish_stopwords,
                    only_alpha=True,
                    split_regex=' ')
    folds_creator_config['data'] = data
    fold_creator = FoldCreator(**folds_creator_config)
    print(data)
    TF_IDF_config["data"] = data
    tf_idf = TF_IDF(**TF_IDF_config)
    tf_idf.create_tf_idf_table((True, True), True)
    tf_idf_table = pd.read_parquet(create_tf_idf_table_to_path).reset_index().to_numpy()
    knn_config["tf_idf_table"] = tf_idf_table

    stratified_kfolds_config['knn_config'] = knn_config
    stratified_kfolds_config['folds_created'] = True
    stratified_kfolds_config['data'] = data
    stratified_kfolds_config['folds'] = fold_creator.folds
    sfkcv = StratifiedKFoldsCV(**stratified_kfolds_config)
    sfkcv.start_knn()


cProfile.run('run_knn()', 'profile_output')
p = pstats.Stats('profile_output')
p.sort_stats(pstats.SortKey.TIME).print_stats(10)
