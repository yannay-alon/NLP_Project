from collections import OrderedDict
from typing import List, Callable


def n_gram(n: int):
    def wrapper(func):
        def extract_features(self, line):

            # Special symbols for the beginning and ending of the line
            if 1 < n:
                start = "(╯°□°）╯︵┻━┻".replace(" ", "").replace("_", "")
                start = f"{start}_{start} " * (n - 1)

                end = "┬─┬ノ(゜-゜ノ)".replace(" ", "").replace("_", "")
                end = f"{end}_{end}"

                line = f"{start}{line[:-1]} {end}"

            split_words = line.split(" ")

            for k in range(1, min(n, len(split_words)) + 1):
                k_grams = zip(*[split_words[i:] for i in range(k)])

                for k_gram in k_grams:
                    split_list = (word_tag.split("_") for word_tag in k_gram)
                    words, tags = list(zip(*split_list))

                    if k == 1:
                        words = words[0]
                        tags = tags[0]

                    func(self, words, tags)

        return extract_features

    return wrapper


class FeatureStatistics:

    def __init__(self, file_path: str):
        self.n_total_features = 0  # Total number of features accumulated
        self.file_path = file_path

        # Init all features dictionaries
        self.capital_tags_dict = OrderedDict()
        self.prefix_tags_dict = OrderedDict()
        self.suffix_tags_dict = OrderedDict()
        self.k_gram_dict = OrderedDict()

        self.dictionaries = [
            self.capital_tags_dict,
            self.prefix_tags_dict,
            self.suffix_tags_dict,
            self.k_gram_dict
        ]
        # TODO: ---Add more count dictionaries here---

        # Extract all features dictionaries
        pairs_dictionaries_functions = [
            self.get_capital_tag_pair_count,
            self.get_prefix_tags_pair_count,
            self.get_suffix_tags_pair_count,
            self.extract_k_grams_test,
        ]

        self.extract_all_pairs(pairs_dictionaries_functions)

    def extract_all_pairs(self, functions: List[Callable]):
        with open(self.file_path) as f:
            for line in f:
                for func in functions:
                    func(line)

    @n_gram(1)
    def get_capital_tag_pair_count(self, cur_word: str, cur_tag: str):
        """
            Extract out of text all word/tag pairs
        """
        capital = cur_word[0].isupper()
        if capital:
            if ("Is_Capital", cur_tag) not in self.capital_tags_dict:
                self.capital_tags_dict[("Is_Capital", cur_tag)] = 0
            self.capital_tags_dict[("Is_Capital", cur_tag)] += 1

    @n_gram(1)
    def get_prefix_tags_pair_count(self, cur_word: str, cur_tag: str):
        max_length = 4
        for length in range(1, min(len(cur_word), max_length) + 1):
            prefix = cur_word[:length]
            if (prefix, cur_tag) not in self.prefix_tags_dict:
                self.prefix_tags_dict[(prefix, cur_tag)] = 0
            self.prefix_tags_dict[(prefix, cur_tag)] += 1

    @n_gram(1)
    def get_suffix_tags_pair_count(self, cur_word: str, cur_tag: str):
        max_length = 4
        for length in range(1, min(len(cur_word), max_length) + 1):
            suffix = cur_word[-length:]
            if (suffix, cur_tag) not in self.suffix_tags_dict:
                self.suffix_tags_dict[(suffix, cur_tag)] = 0
            self.suffix_tags_dict[(suffix, cur_tag)] += 1

    @n_gram(3)
    def extract_k_grams_test(self, words, tags):
        if (words, tags) not in self.k_gram_dict:
            self.k_gram_dict[(words, tags)] = 0
        self.k_gram_dict[(words, tags)] += 1
