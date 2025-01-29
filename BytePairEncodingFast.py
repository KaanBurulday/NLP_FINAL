from collections import defaultdict
import heapq

class BytePairEncoding:
    def __init__(self, text_data, end_of_word_token='<ew>', merges=100):
        """
        :param text_data: Either a list of words or a single string.
        :param end_of_word_token: Special token appended at the end of each word.
        :param merges: Number of BPE merges to perform.
        """
        # If text_data is a single string, split it by whitespace
        if isinstance(text_data, str):
            self.text_data = text_data.strip().split()
        else:
            self.text_data = text_data

        self.end_of_word_token = end_of_word_token
        self.merges = merges

        # Will hold the corpus as a list of token lists, e.g. [["h", "e", "l", "l", "o", "<ew>"], ...]
        self.corpus = []
        # Dictionary to count frequency of each adjacent token pair across entire corpus
        self.pair_counts = defaultdict(int)
        # We build our vocabulary as we go
        self.vocabulary = set()

    def prepare_corpus(self):
        """
        Convert each word into a list of characters + EOW token.
        Example:
            "hello" -> ["h", "e", "l", "l", "o", "<ew>"]
        """
        for word in self.text_data:
            tokens = list(word) + [self.end_of_word_token]
            self.corpus.append(tokens)

    def build_pair_counts(self):
        """
        Initialize pair_counts by counting all adjacent pairs across every word.
        """
        for tokens in self.corpus:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                self.pair_counts[pair] += 1

    def get_max_pair_heap(self):
        """
        Convert the pair_counts dictionary to a max-heap based on frequencies.
        Python's heapq is a min-heap, so we store negative counts.
        :return: a list usable as a heap where each element is (-count, pair).
        """
        heap = []
        for pair, freq in self.pair_counts.items():
            # Push (-frequency, pair) so the most frequent pair is on top
            heapq.heappush(heap, (-freq, pair))
        return heap

    def merge_pair_in_corpus(self, pair):
        """
        Merge the given pair (x, y) -> "xy" across the entire corpus.
        Update pair_counts locally, rather than recalculating from scratch.
        """
        merged_token = ''.join(pair)  # e.g., ("h", "e") -> "he"

        for tokens in self.corpus:
            i = 0
            # We'll scan through each word once, merging in-place
            while i < len(tokens) - 1:
                current_pair = (tokens[i], tokens[i + 1])
                if current_pair == pair:
                    # "Left" and "Right" tokens around the pair (if any)
                    left_token = tokens[i - 1] if i - 1 >= 0 else None
                    right_token = tokens[i + 2] if i + 2 < len(tokens) else None

                    # Decrement counts for old pairs
                    # Pair to remove: (left_token, tokens[i]) if left_token exists
                    if left_token is not None:
                        self.pair_counts[(left_token, tokens[i])] -= 1
                    # Pair to remove: (tokens[i], tokens[i+1]) which is current_pair
                    self.pair_counts[current_pair] -= 1
                    # Pair to remove: (tokens[i+1], right_token) if right_token exists
                    if right_token is not None:
                        self.pair_counts[(tokens[i + 1], right_token)] -= 1

                    # Merge the pair into a single token
                    tokens[i] = merged_token
                    del tokens[i + 1]

                    # Increment counts for newly formed pairs
                    # (left_token, merged_token) and (merged_token, right_token)
                    if left_token is not None:
                        self.pair_counts[(left_token, merged_token)] += 1
                    if right_token is not None:
                        self.pair_counts[(merged_token, right_token)] += 1

                    # Don't advance i, because we need to check
                    # the new pair (merged_token, next_token) if it merges again
                else:
                    i += 1

    def run_bpe(self, show_counter: bool):
        """
        Run the entire BPE pipeline:
            1) Prepare the corpus (list of token-lists).
            2) Build initial pair counts.
            3) For 'merges' iterations, pick the most frequent pair and merge it.
            4) Return the final vocabulary and the final corpus as strings.
        """
        # Step 1: Prepare corpus
        self.prepare_corpus()

        # Step 2: Build pair counts
        self.build_pair_counts()

        # Initialize vocabulary with all characters (tokens) currently in the corpus
        for tokens in self.corpus:
            self.vocabulary.update(tokens)

        # Step 3: Perform merges
        count = 0
        for iteration in range(self.merges):
            if show_counter:
                if iteration % 1000 == 0:
                    print(f"{count}000 iteration is done.")
                    count += 1
            # Build a max-heap to get the pair with the highest frequency
            heap = self.get_max_pair_heap()
            if not heap:
                # No pairs left at all
                break

            # Get the most frequent pair
            freq, pair = heapq.heappop(heap)
            freq = -freq  # Convert back to positive

            # If the most frequent pair has a frequency of 0, we're done
            if freq <= 0:
                break

            # Merge this pair in the corpus
            self.merge_pair_in_corpus(pair)

            # Add the newly formed token to the vocabulary
            new_token = ''.join(pair)
            self.vocabulary.add(new_token)

        # Step 4: Return results
        # Convert token lists back to strings if desired
        final_corpus = [' '.join(tokens) for tokens in self.corpus]

        # Sort the vocabulary for a consistent output
        sorted_vocab = sorted(tok for tok in self.vocabulary if tok)

        return sorted_vocab, final_corpus
