from collections import OrderedDict
from typing import Iterable, List
from .History import History
from .Key import Key
import re


def polynomial_threshold(coefficients: List[float]):
    """
    Creates a threshold as a function of the number of elements in the k-gram

    :param coefficients: The coefficients for the polynomial function
    :return: A function that takes the number of elements in the k-gram and returns the threshold
    """

    def threshold(length: int):
        total = 0
        for index, coefficient in enumerate(coefficients):
            total += coefficient * length ** index
        return int(max(1, total))

    return threshold


class FeatureStatistics:
    """
    Gets the feature for the given histories
    """

    Has_Pun_Threshold: int = polynomial_threshold([25])
    Alphanum_Threshold: int = polynomial_threshold([25])
    Has_num_Threshold: int = polynomial_threshold([20])
    All_num_Threshold: int = polynomial_threshold([15])
    Start_Capital_Threshold: int = polynomial_threshold([25])
    All_Capital_Threshold: int = polynomial_threshold([20])
    Prefix_Threshold: int = polynomial_threshold([20])
    Suffix_Threshold: int = polynomial_threshold([15])
    n_gram_Threshold: int = polynomial_threshold([24, -10, 1])
    length_Threshold: int = polynomial_threshold([20, 3, -0.2])

    def __init__(self, histories: Iterable["History"]):
        self.feature_dictionary = OrderedDict()

        self.feature_functions = [
            create_capital_features,
            create_prefix_features,
            create_suffix_features,
            create_n_gram_features,
            create_alpha_num_features,
            create_length_features,
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
            # TODO may has the starter and finisher as keys, need to know the starter and finisher somehow and then
            #  solve this
            for key in keys:
                yield key
        # raise StopIteration

    # </editor-fold>


# TODO: add more feature extractors:
#  capital in word (not just the first letter)

# <editor-fold desc="create_features functions">
def create_capital_features(history: "History") -> Iterable["Key"]:
    """
    Extract features where the current word starts with a capital with a specific tag
    """
    cur_word = history.words[-1]
    cur_tag = history.tags[-1]
    capital = cur_word[0].isupper()  # Check if the first letter is a capital
    if capital:
        key = Key(("Starts_Capital",), (cur_tag,), FeatureStatistics.Start_Capital_Threshold)
        yield key
    capital = cur_word.isupper()  # Check if the all of the letters is a capital
    if capital:
        key = Key(("All_Capital",), (cur_tag,), FeatureStatistics.All_Capital_Threshold)
        yield key
    # raise StopIteration


def create_length_features(history: "History") -> Iterable["Key"]:
    """
    Extract features where the current word starts with a capital with a specific tag
    """
    cur_word = history.words[-1]
    cur_tag = history.tags[-1]
    lengthes = [1, 2, 4, 6, 8, 10, 12, 14]
    lowerbound = 1
    word_len = len(cur_word)
    for upper_bound in lengthes:
        if word_len <= upper_bound:
            key = Key((f"{lowerbound}-{upper_bound}Length",), (cur_tag,), FeatureStatistics.length_Threshold)
            yield key
            break
        lowerbound = upper_bound
    key = Key((f"{lowerbound}-{upper_bound}Length",), (cur_tag,), FeatureStatistics.length_Threshold)
    yield key
    # raise StopIteration


def create_prefix_features(history: "History") -> Iterable["Key"]:
    """
    Extract features for the prefix of the current word with a specific tag
    """
    cur_word = history.words[-1]
    cur_tag = history.tags[-1]
    max_length = 4
    for length in range(1, min(len(cur_word), max_length) + 1):
        # Indicates a prefix feature
        prefix = f"PREFIX_{cur_word[:length]}"
        key = Key((prefix,), (cur_tag,), FeatureStatistics.Prefix_Threshold)
        yield key
    # raise StopIteration


def create_suffix_features(history: "History") -> Iterable["Key"]:
    """
    Extract features for the suffix of the current word with a specific tag
    """
    cur_word = history.words[-1]
    cur_tag = history.tags[-1]
    max_length = 4
    for length in range(1, min(len(cur_word), max_length) + 1):
        # Indicates a suffix feature
        suffix = f"SUFFIX_{cur_word[-length:]}"
        key = Key((suffix,), (cur_tag,), FeatureStatistics.Suffix_Threshold)
        yield key
    # raise StopIteration


def create_alpha_num_features(history: "History") -> Iterable["Key"]:
    """
    Extract features where the current word starts with a capital with a specific tag
    """
    cur_word = history.words[-1]
    cur_tag = history.tags[-1]
    if cur_word != re.sub(r"[0-9]+", "", cur_word):
        key = Key(("Has_Num",), (cur_tag,), FeatureStatistics.Has_num_Threshold)
        yield key
    all_num = cur_word.replace(",", "").replace(".", "").isnumeric()
    if re.fullmatch(r"[0-9,.]+", cur_word):
        key = Key(("All_num",), (cur_tag,), FeatureStatistics.All_num_Threshold)
        yield key
    if cur_word.isalnum():
        key = Key(("Is_alnum",), (cur_tag,), FeatureStatistics.Alphanum_Threshold)
        yield key
    if cur_word != re.sub(r'[^\w\s]', '', cur_word):
        key = Key(("Has_pun",), (cur_tag,), FeatureStatistics.Has_Pun_Threshold)
        yield key
    # raise StopIteration


def create_n_gram_features(history: "History") -> Iterable["Key"]:
    """
    Extract features for the k-grams with their corresponding tags
    """
    for length in range(1, len(history) + 1):
        words = history.words[-length:]
        tags = history.tags[-length:]
        key = Key(words, tags, FeatureStatistics.n_gram_Threshold)
        yield key
    # raise StopIteration
# </editor-fold>
