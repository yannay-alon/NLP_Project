from collections import OrderedDict
from typing import Iterable, List, Callable
from Project.FeatureExtraction.History import History
from Project.FeatureExtraction.Key import Key
import re


def polynomial_threshold(coefficients: List[float]) -> Callable[[int], int]:
    """
    Creates a threshold as a function of the number of elements in the k-gram

    :param coefficients: The coefficients for the polynomial function
    :return: A function that takes the number of elements in the k-gram and returns the threshold
    """

    def threshold(length: int) -> int:
        total = 0
        for index, coefficient in enumerate(coefficients):
            total += coefficient * length ** index
        return int(max(1, total))

    return threshold


class FeatureStatistics:
    """
    Gets the feature for the given histories
    """

    # <editor-fold desc="Thresholds">
    Has_Pun_Threshold = polynomial_threshold([25])
    Alphanum_Threshold = polynomial_threshold([25])
    Has_num_Threshold = polynomial_threshold([20])
    All_num_Threshold = polynomial_threshold([15])
    Start_Capital_Threshold = polynomial_threshold([25])
    All_Capital_Threshold = polynomial_threshold([20])
    Prefix_Threshold = polynomial_threshold([20, -7, 0.7])
    Suffix_Threshold = polynomial_threshold([15, -5, 0.5])
    n_gram_Threshold = polynomial_threshold([24, -10, 1])
    length_Threshold = polynomial_threshold([20, 2, -0.2])

    # </editor-fold>

    def __init__(self, histories: Iterable["History"]):
        self.feature_dictionary = OrderedDict()

        self.feature_functions = [
            FeatureStatistics.create_capital_features,
            FeatureStatistics.create_prefix_features,
            FeatureStatistics.create_suffix_features,
            FeatureStatistics.create_n_gram_features,
            FeatureStatistics.create_alpha_num_features,
            FeatureStatistics.create_length_features,
        ]

        # Create all relevant features
        for history in histories:
            self.initialize_feature_dictionary(history)

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

    # <editor-fold desc="create_features functions">
    @staticmethod
    def create_capital_features(history: "History") -> Iterable["Key"]:
        """
        Extract features where the current word starts with a capital with a specific tag
        """
        cur_word = history.words[-1]
        cur_tag = history.tags[-1]
        capital = cur_word[0].isupper()  # Check if the first letter is a capital
        if capital:
            key = Key(("Starts_Capital",), (cur_tag,), FeatureStatistics.Start_Capital_Threshold(1))
            yield key
        capital = cur_word.isupper()  # Check if the all of the letters is a capital
        if capital:
            key = Key(("All_Capital",), (cur_tag,), FeatureStatistics.All_Capital_Threshold(1))
            yield key

    @staticmethod
    def create_length_features(history: "History") -> Iterable["Key"]:
        """
        Extract features where the current word starts with a capital with a specific tag
        """
        cur_word = history.words[-1]
        cur_tag = history.tags[-1]
        lengths = [1, 2, 4, 6, 8, 10, 12, 14]
        lower_bound = lengths[0]
        word_len = len(cur_word)
        for upper_bound in lengths:
            if word_len <= upper_bound:
                key = Key((f"Length_{lower_bound}-{upper_bound}",),
                          (cur_tag,), FeatureStatistics.length_Threshold(word_len))
                yield key
                break
            lower_bound = upper_bound
        key = Key((f"Length_>{lengths[-1]}",), (cur_tag,), FeatureStatistics.length_Threshold(word_len))
        yield key

    @staticmethod
    def create_prefix_features(history: "History") -> Iterable["Key"]:
        """
        Extract features for the prefix of the current word with a specific tag
        """
        cur_word = history.words[-1]
        cur_tag = history.tags[-1]
        max_length = 4
        for prefix_length in range(1, min(len(cur_word), max_length) + 1):
            # Indicates a prefix feature
            prefix = f"PREFIX_{cur_word[:prefix_length]}"
            key = Key((prefix,), (cur_tag,), FeatureStatistics.Prefix_Threshold(prefix_length))
            yield key

    @staticmethod
    def create_suffix_features(history: "History") -> Iterable["Key"]:
        """
        Extract features for the suffix of the current word with a specific tag
        """
        cur_word = history.words[-1]
        cur_tag = history.tags[-1]
        max_length = 4
        for suffix_length in range(1, min(len(cur_word), max_length) + 1):
            # Indicates a suffix feature
            suffix = f"SUFFIX_{cur_word[-suffix_length:]}"
            key = Key((suffix,), (cur_tag,), FeatureStatistics.Suffix_Threshold(suffix_length))
            yield key

    @staticmethod
    def create_alpha_num_features(history: "History") -> Iterable["Key"]:
        """
        Extract features where the current word starts with a capital with a specific tag
        """
        cur_word = history.words[-1]
        cur_tag = history.tags[-1]
        if re.search(r"\d+", cur_word) is not None:
            key = Key(("Has_Num",), (cur_tag,), FeatureStatistics.Has_num_Threshold(1))
            yield key

        if re.fullmatch(r"[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?", cur_word) is not None:
            key = Key(("All_num",), (cur_tag,), FeatureStatistics.All_num_Threshold(1))
            yield key
        if cur_word.isalnum():
            key = Key(("Is_alnum",), (cur_tag,), FeatureStatistics.Alphanum_Threshold(1))
            yield key
        if re.search(r"[^\w\s]", cur_word) is not None:
            key = Key(("Has_pun",), (cur_tag,), FeatureStatistics.Has_Pun_Threshold(1))
            yield key

    @staticmethod
    def create_n_gram_features(history: "History") -> Iterable["Key"]:
        """
        Extract features for the k-grams with their corresponding tags
        """
        for length in range(1, len(history) + 1):
            words = history.words[-length:]
            tags = history.tags[-length:]
            key = Key(words, tags, FeatureStatistics.n_gram_Threshold(length))
            yield key

    # </editor-fold>
