from collections import OrderedDict
from typing import List, Callable
from .TextEditor import TextEditor


def n_gram(n: int):
    def wrapper(func):
        def extract_features(self, line, decorated_line):
            if n == 1:
                split_words = line.split(" ")
            else:
                split_words = decorated_line.split(" ")

            for k in range(1, min(n, len(split_words)) + 1):
                k_grams = zip(*[split_words[i:] for i in range(k)])

                for k_gram in k_grams:
                    split_list = (word_tag.split("_") for word_tag in k_gram)
                    words, tags = list(zip(*split_list))

                    if n == 1:
                        words = words[0]
                        tags = tags[0]

                    func(self, words, tags)

        return extract_features

    return wrapper


class FeatureStatistics:
    max_gram = 3

    def __init__(self, text_editor: "TextEditor"):
        self.text_editor = text_editor
        FeatureStatistics.max_gram = text_editor.window_size

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

        # Extract all features dictionaries
        pairs_dictionaries_functions = [
            self.get_capital_tag_pair_count,
            self.get_prefix_tags_pair_count,
            self.get_suffix_tags_pair_count,
            self.extract_k_grams,
        ]

        self.extract_all(pairs_dictionaries_functions)

    # <editor-fold desc="Extraction of features">

    def extract_all(self, functions: List[Callable]):
        for line, decorated_line in self.text_editor.read_file():
            for func in functions:
                func(line, decorated_line)

    @n_gram(1)
    def get_capital_tag_pair_count(self, cur_word: str, cur_tag: str):
        """
            Extract out of text all word/tag pairs
        """
        capital = cur_word[0].isupper()
        if capital:
            key = (("Is_Capital",), (cur_tag,))
            if key not in self.capital_tags_dict:
                self.capital_tags_dict[key] = 0
            self.capital_tags_dict[key] += 1

    @n_gram(1)
    def get_prefix_tags_pair_count(self, cur_word: str, cur_tag: str):
        max_length = 4
        for length in range(1, min(len(cur_word), max_length) + 1):
            prefix = f"PREFIX_{cur_word[:length]}"
            key = ((prefix,), (cur_tag,))
            if key not in self.prefix_tags_dict:
                self.prefix_tags_dict[key] = 0
            self.prefix_tags_dict[key] += 1

    @n_gram(1)
    def get_suffix_tags_pair_count(self, cur_word: str, cur_tag: str):
        max_length = 4
        for length in range(1, min(len(cur_word), max_length) + 1):
            suffix = f"SUFFIX_{cur_word[-length:]}"
            key = ((suffix,), (cur_tag,))
            if key not in self.suffix_tags_dict:
                self.suffix_tags_dict[key] = 0
            self.suffix_tags_dict[key] += 1

    @n_gram(max_gram)
    def extract_k_grams(self, words, tags):
        if (words, tags) not in self.k_gram_dict:
            self.k_gram_dict[(words, tags)] = 0
        self.k_gram_dict[(words, tags)] += 1

    # </editor-fold>
