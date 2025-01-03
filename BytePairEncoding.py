from collections import Counter


class BytePairEncoding:
    def __init__(self, text_data: list[str] | str, end_of_word_token: str, n: int):
        self.end_of_word_token = end_of_word_token
        self.corpus = text_data if isinstance(text_data, list) else text_data.split()
        self.n = n

    def prepare_corpus(self):
        return [" ".join(list(word) + [self.end_of_word_token]) for word in self.corpus]

    def get_initial_vocabulary(self, corpus):
        vocab = set("".join(corpus))
        return sorted(vocab)

    def find_most_frequent_pair(self, corpus):
        pairs = Counter()
        for word in corpus:
            tokens = word.split()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] += 1
        return max(pairs, key=pairs.get, default=None), pairs

    def replace_pair(self, pair, corpus):
        new_corpus = []
        pair_str = " ".join(pair)
        for word in corpus:
            new_word = word.replace(pair_str, "".join(pair))
            new_corpus.append(new_word)
        return new_corpus

    def BPE(self, show_counter: bool, make_corpus_unique: bool):
        corpus = self.prepare_corpus()
        vocabulary = self.get_initial_vocabulary(corpus)

        count = 0
        for _ in range(self.n):
            if show_counter:
                if _ % 1000 == 0:
                    print(f"{count}000 iteration is done.")
                    count += 1
            most_frequent_pair, pairs = self.find_most_frequent_pair(corpus)
            if not most_frequent_pair or pairs[most_frequent_pair] == 0:
                break
            corpus = self.replace_pair(most_frequent_pair, corpus)
            new_token = "".join(most_frequent_pair)
            if new_token not in vocabulary:
                vocabulary.append(new_token)

        if make_corpus_unique:
            corpus_word_counter = Counter(corpus)
            for word in corpus_word_counter:
                if corpus_word_counter[word] > 1:
                    for _ in range(corpus_word_counter[word] - 1):
                        corpus.remove(word)

        if '' in vocabulary:
            vocabulary.remove('')
        if ' ' in vocabulary:
            vocabulary.remove(' ')

        return vocabulary, corpus

    def wordCount(self, data):
        pairs = Counter()
        for word in data:
            pairs[word] += 1
        return pairs
