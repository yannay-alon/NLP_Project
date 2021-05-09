from collections import OrderedDict
from typing import Iterable
from .History import History
from .Key import Key


class FeatureStatistics:
    """
    Gets the feature for the given text
    """

    def __init__(self, histories: Iterable["History"]):
        self.feature_dictionary = OrderedDict()

        self.feature_functions = [
            self.create_capital_features,
            self.create_prefix_features,
            self.create_suffix_features,
            self.create_n_gram_features,
        ]

        # Create all relevant features
        for history in histories:
            self.initialize_feature_dictionary(history)

    # <editor-fold desc="Extraction of features">

    def initialize_feature_dictionary(self, history: "History") -> None:
        """
        Call all of the feature-extraction function
        """
        for key in self.get_keys(history):
            if key not in self.feature_dictionary:
                self.feature_dictionary[key] = 0
            self.feature_dictionary[key] += 1

    def get_keys(self, history: "History") -> Iterable["Key"]:
        for func in self.feature_functions:
            keys = func(history)
            for key in keys:
                yield key
        raise StopIteration

    # TODO: add more feature extractors:
    #  digit detector - word contains digit
    #  capital in word (not just the first letter)

    def create_capital_features(self, history: "History") -> Iterable["Key"]:
        """
        Extract features where the current word starts with a capital with a specific tag
        """
        cur_word = history.words[-1]
        cur_tag = history.tags[-1]
        capital = cur_word[0].isupper()  # Check if the first letter is a capital
        if capital:
            key = Key(("Starts_Capital",), (cur_tag,))
            yield [key]
        capital = cur_word.isupper()  # Check if the all of the letters is a capital
        if capital:
            key = Key(("All_Capital",), (cur_tag,))
            yield [key]
        raise StopIteration

    def create_prefix_features(self, history: "History") -> Iterable["Key"]:
        """
        Extract features for the prefix of the current word with a specific tag
        """
        cur_word = history.words[-1]
        cur_tag = history.tags[-1]
        max_length = 4
        for length in range(1, min(len(cur_word), max_length) + 1):
            # Indicates a prefix feature
            prefix = f"PREFIX_{cur_word[:length]}"
            key = ((prefix,), (cur_tag,))
            yield key
        raise StopIteration

    def create_suffix_features(self, history: "History") -> Iterable["Key"]:
        """
        Extract features for the suffix of the current word with a specific tag
        """
        cur_word = history.words[-1]
        cur_tag = history.tags[-1]
        max_length = 4
        for length in range(1, min(len(cur_word), max_length) + 1):
            # Indicates a suffix feature
            suffix = f"SUFFIX_{cur_word[-length:]}"
            key = ((suffix,), (cur_tag,))
            yield key
        raise StopIteration

    def create_n_gram_features(self, history: "History") -> Iterable["Key"]:
        """
        Extract features for the k-grams with their corresponding tags
        """
        for length in range(1, len(history) + 1):
            words = history.words[-length:]
            tags = history.tags[-length:]
            key = (words, tags)
            yield key
        raise StopIteration

    # </editor-fold>
