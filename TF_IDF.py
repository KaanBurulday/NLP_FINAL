import pathlib
import os
from collections import Counter
from math import log10
import nltk
import snowballstemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import BytePairEncoding
from os.path import isfile


# TF-IDF(i, j) = TF(i, j) * IDF(i, j)
# TF(i, j) = frequency (count) of i in document j / total number of terms in j
# IDF(i, j) = log(Total number of documents / The number of documents which term t appears)

# Will contain the document name, terms and their counts, total number of terms and the class of the document
# as { "Doc1": { "terms": { "termA": 1, "termB":3 }, "termCount": 4, "class": 1}}. Will be used for TF calculation.

# a counter for each term. Will be used for IDF calculation for The number of documents which term t appears

def sanitizer_char(text: str | list[str], stop_words: list[str], only_alpha: bool, split_regex: str) -> str:
    word_list = text if isinstance(text, list) else text.split(split_regex)
    sanitized_text = ""
    if only_alpha:
        for word in word_list:
            for char in word:
                if char not in stop_words and char.isalpha():
                    sanitized_text += char
            sanitized_text += " "
    else:
        for word in word_list:
            for char in word:
                if char not in stop_words:
                    sanitized_text += char
            sanitized_text += " "
    return sanitized_text


def sanitizer_string(text: str | list[str], stop_words: list[str], only_alpha: bool, split_regex: str) -> str:
    word_list = text if isinstance(text, list) else text.split(split_regex)
    sanitized_text = ""
    if only_alpha:
        for word in word_list:
            if word not in stop_words and word.isalpha():
                sanitized_text += word + " "
    else:
        for word in word_list:
            if word not in stop_words:
                sanitized_text += word + " "
    return sanitized_text


class TF_IDF:
    def __init__(self, **kwargs):
        self.data_path = kwargs.get("data_path", f"{pathlib.Path().resolve()}\\raw_texts")
        self.corpus_path = kwargs.get("corpus_path", f"{pathlib.Path().resolve()}\\corpus.txt")
        self.vocab_path = kwargs.get("vocab_path", f"{pathlib.Path().resolve()}\\vocab.txt")
        self.stop_words: list[str] = kwargs.get("stop_words", [''])
        self.word_split_regex = kwargs.get("word_split_regex", ' ')
        self.only_alpha = kwargs.get("only_alpha", True)
        self.end_of_word_token = kwargs.get("end_of_word_token", "_")
        self.n = kwargs.get("n", 10000)
        self.show_bpe_counter = kwargs.get("show_bpe_counter", True)
        self.make_corpus_unique = kwargs.get("make_corpus_unique", True)

        self.create_tf_idf_table_to_path = kwargs.get("create_tf_idf_table_to_path",
                                                      f"{pathlib.Path().resolve()}\\TF_IDF_Table.csv")
        self.use_nltk_stemmer = kwargs.get("use_nltk_stemmer", False)

        self.vocabulary: list[str] = []
        self.corpus: list[str] = []
        self.documents = {}
        self.total_number_of_documents = 0
        self.total_term_counter = Counter()
        self.tf_idf_table = {}

    def initialize_documents(self):
        # to keep tract of the document names, commented out due to some performance hits with stratified k-folds algorithm
        # directory_names = os.listdir(self.data_path)
        # for directory_name in directory_names:
        #     file_names = os.listdir(os.path.join(self.data_path, directory_name))
        #     self.total_number_of_documents += len(file_names)
        #     for file_name in file_names:
        #         self.documents[f"Doc{file_name.split('.')[0]}"] = {"terms": {}, "term_count": 0,
        #                                                            "class": directory_name}
        #         self.tf_idf_table[f"Doc{file_name.split('.')[0]}"] = {}
        document_count = 0
        directory_names = os.listdir(self.data_path)
        for directory_name in directory_names:
            file_names = os.listdir(os.path.join(self.data_path, directory_name))
            self.total_number_of_documents += len(file_names)
            for file_name in file_names:
                document_count += 1
                self.documents[f"Doc{document_count}"] = {"terms": {}, "term_count": 0,
                                                          "class": directory_name}
                self.tf_idf_table[f"Doc{document_count}"] = {}

    def get_total_text(self):
        total_text = ""
        directory_names = os.listdir(self.data_path)
        for directory_name in directory_names:
            file_names = os.listdir(os.path.join(self.data_path, directory_name))
            for file_name in file_names:
                file_path = os.path.join(self.data_path, directory_name, file_name)
                with(open(file_path, 'r')) as file:
                    sanitized_text = sanitizer_string(file.read(), self.stop_words, self.only_alpha,
                                                      self.word_split_regex)
                    total_text += sanitized_text.lower() + " "
        return total_text

    def create_corpus_vocabulary_bpe(self):
        total_text = self.get_total_text()
        bpe = BytePairEncoding.BytePairEncoding(text_data=total_text, end_of_word_token=self.end_of_word_token,
                                                n=self.n)
        bpe_output = bpe.BPE(self.show_bpe_counter, self.make_corpus_unique)
        if self.use_nltk_stemmer:
            turkish_stemmer = snowballstemmer.stemmer('turkish')

            # vocabulary = [word[:-1] if word[len(word) - 1] == end_of_word_token else word for word in bpe_output[0]]
            self.vocabulary = [turkish_stemmer.stemWord(word) for word in bpe_output[0]]
            self.corpus = [turkish_stemmer.stemWord(word[:-1]) if word[
                                                                      len(word) - 1] == self.end_of_word_token else turkish_stemmer.stemWord(
                word) for word in bpe_output[1]]
        else:
            self.vocabulary = [word for word in bpe_output[0]]
            self.corpus = [word[:-1] if word[len(word) - 1] == self.end_of_word_token else word for word in
                           bpe_output[1]]

    def create_tokens_with_nltk(self):
        total_text = self.get_total_text()

        tokens = word_tokenize(total_text, language="turkish")
        turkish_stemmer = snowballstemmer.stemmer('turkish')

        # Remove stopwords
        nltk.download('stopwords')  # Ensure stopwords are downloaded
        turkish_stopwords = set(stopwords.words('turkish'))
        turkish_stopwords_concatenated = set(turkish_stopwords).union(set(self.stop_words))
        if self.use_nltk_stemmer:
            filtered_tokens = [
                turkish_stemmer.stemWord(word)
                for word in tokens if word.lower() not in turkish_stopwords_concatenated
            ]
        else:
            filtered_tokens = [
                word for word in tokens if word.lower() not in turkish_stopwords_concatenated
            ]

        if '' in filtered_tokens:
            filtered_tokens.remove('')

        unique_tokens = list(set(filtered_tokens)) if self.make_corpus_unique else filtered_tokens
        self.vocabulary = unique_tokens
        self.corpus = filtered_tokens if not self.make_corpus_unique else unique_tokens

    def fill_terms_of_documents(self, split_regex: str):
        document_count = 0
        directory_names = os.listdir(self.data_path)
        for directory_name in directory_names:
            file_names = os.listdir(os.path.join(self.data_path, directory_name))
            for file_name in file_names:
                term_counter = Counter()
                file_path = os.path.join(self.data_path, directory_name, file_name)
                with(open(file_path)) as file:
                    for line in file.readlines():
                        sanitized_text = sanitizer_string(line, self.stop_words, self.only_alpha, split_regex)
                        for word in sanitized_text.split(split_regex):
                            if word in self.corpus:
                                term_counter[word] += 1
                                if self.total_term_counter[word] == 0:
                                    self.total_term_counter[word] += 1

                document_count += 1
                for term in term_counter:
                    self.documents[f"Doc{document_count}"]["terms"][term] = term_counter[term]
                self.documents[f"Doc{document_count}"]["term_count"] = sum(term_counter.values())

    def fill_corpus_vocabulary_by_file(self):
        with open(self.corpus_path, "r", encoding="utf-8") as file:
            for line in file.readlines():
                self.corpus.append(line.strip())

        with open(self.vocab_path, "r", encoding="utf-8") as file:
            for line in file.readlines():
                self.vocabulary.append(line.strip())

    def create_corpus_vocab_file(self):
        with open(self.corpus_path, "w", encoding="utf-8") as file:
            for i, word in enumerate(self.corpus):
                if i == len(self.corpus) - 1:
                    file.write(f"{word}")
                else:
                    file.write(f"{word}\n")

        with open(self.vocab_path, "w", encoding="utf-8") as file:
            for i, word in enumerate(self.vocabulary):
                if i == len(self.vocabulary) - 1:
                    file.write(f"{word}")
                else:
                    file.write(f"{word}\n")

    def create_tf_idf_table(self, use_files: (bool, bool), use_nltk: bool):
        """use_files: (true for using them, true for recreating them)"""
        self.initialize_documents()

        if use_files[0]:
            if use_nltk:
                self.create_tokens_with_nltk()
            else:
                self.create_corpus_vocabulary_bpe()

            if use_files[1] or (not isfile(self.corpus_path) or not isfile(self.vocab_path)):
                self.create_corpus_vocab_file()
            else:
                self.fill_corpus_vocabulary_by_file()
        else:
            if use_nltk:
                self.create_tokens_with_nltk()
            else:
                self.create_corpus_vocabulary_bpe()

        self.fill_terms_of_documents(self.word_split_regex)

        sorted_document_names = sorted(self.documents.keys(), key=lambda x: int(x[3:]))
        for document_name in sorted_document_names:
            for term in self.corpus:
                if term in self.documents[document_name]["terms"]:
                    tf = self.documents[document_name]["terms"][term] / self.documents[document_name]["term_count"]
                else:
                    tf = 0
                idf = log10(self.total_number_of_documents / (1 + self.total_term_counter[term]))
                tf_idf = tf * idf
                self.tf_idf_table[document_name][term] = tf_idf
            self.tf_idf_table[document_name]["class"] = self.documents[document_name]["class"]

        df = pd.DataFrame.from_dict(self.tf_idf_table, orient='index')
        df = df.loc[sorted_document_names]
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Doc'}, inplace=True)
        columns = [col for col in df.columns if col != 'class'] + ['class']
        df = df[columns]
        df.to_csv(self.create_tf_idf_table_to_path, index=False)

        return df
