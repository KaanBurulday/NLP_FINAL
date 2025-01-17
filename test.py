import pathlib
import time
from pstats import SortKey

import pandas

from FoldCreator import FoldCreator
from TF_IDF_V2 import TF_IDF

assignment_data_path = f"{pathlib.Path().resolve()}\\makaleler-yazarlar"
#assignment_data_path = f"{pathlib.Path().resolve()}\\raw_texts"
turkish_stopwords = [
    "acaba", "ama", "ancak", "aslında", "az", "bazı", "belki", "biri", "birkaç", "birşey",
    "biz", "bu", "çok", "çünkü", "da", "daha", "de", "defa", "diye", "eğer", "en", "gibi",
    "hem", "hep", "hepsi", "her", "hiç", "ile", "ise", "kez", "ki", "kim", "mı", "mu",
    "mü", "nasıl", "ne", "neden", "nerde", "nerede", "nereye", "niçin", "niye", "o",
    "sanki", "şey", "siz", "şu", "tüm", "ve", "veya", "ya", "yani"
]

config = {
    'data_base_path': assignment_data_path,
    'K': 10,
    'stop_words': turkish_stopwords,
    'only_alpha': True,
    'split_regex': ' ',
    'use_sklearn': True
}

create_tf_idf_table_to_path = f"{pathlib.Path().resolve()}\\TF_IDF_V2_Table.parquet"
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
}

use_numpy = True
start_time = time.time()

fold_creator = FoldCreator(**config)
fold_creator.save_folds()

TF_IDF_config["data"] = fold_creator.data

tf_idf = TF_IDF(**TF_IDF_config)

import cProfile
import pstats

def test():
    df = tf_idf.create_tf_idf_table((True, True), True)
    df_np = df.to_numpy()
    print(df_np[0])
    print(df.head(10))

cProfile.run('test()', 'profile_output')
p = pstats.Stats('profile_output')
p.sort_stats(SortKey.TIME).print_stats(10)
#tf_idf.create_tf_idf_table((True, True), True)

print(f"\nTF-IDF Table Creation Duration:{time.time() - start_time}s")

# print(skfcv.class_data_count_per_fold)
#
# sum = 0
# for key in skfcv.class_data_count_per_fold:
#     sum += skfcv.class_data_count_per_fold[key]
# print(sum)

#print(fold_creator.data)

#fold_creator.save_folds()


# from sklearn.model_selection import StratifiedKFold
#
# skf = StratifiedKFold(n_splits=10, shuffle=True)
# X = fold_creator.data['text']
# y = fold_creator.data['class']
# for i, (train_index, test_index) in enumerate(skf.split(X, y)):
#     fold_creator.folds[i] = fold_creator.data.iloc[test_index]
#
# print(fold_creator.folds)
