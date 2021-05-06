from collections import OrderedDict
from typing import List, Callable
from .TextEditor import TextEditor


def n_gram(n: int):
    """
    A decorator for a n-gram \n
    Reads n-grams from the line and call the function iteratively

    :param n: the maximal number of element in the n-gram
    :return: The wrapper function
    """

    def wrapper(func: Callable[["FeatureStatistics", str, str], None]):
        def extract_features(self: "FeatureStatistics", line: str, decorated_line: str) -> None:

            # Split the line into word-tag pairs
            if n == 1:
                split_words = line.split(" ")
            else:
                split_words = decorated_line.split(" ")

            for k in range(1, min(n, len(split_words)) + 1):
                # Creates the k grams from the word-tag pairs
                k_grams = zip(*[split_words[i:] for i in range(k)])

                # Iteratively call the function with the current k-gram
                for k_gram in k_grams:
                    # Split the word-tag grams into a tuple of words and a tuple of tags
                    split_list = (word_tag.split("_") for word_tag in k_gram)
                    words, tags = list(zip(*split_list))

                    # unpack tuples of size 1
                    if len(words) == 1:
                        words = words[0]
                        tags = tags[0]

                    # Call the function with the k-gram
                    func(self, words, tags)

        return extract_features

    return wrapper


class FeatureStatistics:
    """
    Gets the feature for the given text
    """

    # The maximal length of n-grams
    max_gram = 3

    def __init__(self, text_editor: "TextEditor"):
        self.text_editor = text_editor
        FeatureStatistics.max_gram = text_editor.window_size

        # Initialize all features dictionaries
        self.capital_tags_dict = OrderedDict()
        self.prefix_tags_dict = OrderedDict()
        self.suffix_tags_dict = OrderedDict()
        self.k_gram_dict = OrderedDict()

        # The dictionaries to store the features
        self.dictionaries = [
            self.capital_tags_dict,
            self.prefix_tags_dict,
            self.suffix_tags_dict,
            self.k_gram_dict
        ]

        # The functions to extract the features
        dictionaries_functions = [
            self.get_capital_tag_pair_count,
            self.get_prefix_tags_pair_count,
            self.get_suffix_tags_pair_count,
            self.extract_k_grams,
        ]

        # Extract all features dictionaries
        self.extract_all(dictionaries_functions)

    # <editor-fold desc="Extraction of features">

    def extract_all(self, functions: List[Callable]) -> None:
        """
        Call all of the feature-extraction function

        :param functions: A list of all feature-extraction functions
        """
        for line, decorated_line in self.text_editor.read_file():
            for func in functions:
                func(line, decorated_line)

    @n_gram(1)
    def get_capital_tag_pair_count(self, cur_word: str, cur_tag: str) -> None:
        """
        Extract features where the current word has a capital with a specific tag

        :param cur_word: The current word
        :param cur_tag: The current tag
        """
        capital = cur_word[0].isupper()
        if capital:
            key = (("Is_Capital",), (cur_tag,))
            if key not in self.capital_tags_dict:
                self.capital_tags_dict[key] = 0
            self.capital_tags_dict[key] += 1

    @n_gram(1)
    def get_prefix_tags_pair_count(self, cur_word: str, cur_tag: str) -> None:
        """
        Extract features for the prefix of the current word with a specific tag

        :param cur_word: The current word
        :param cur_tag: The current tag
        """
        max_length = 4
        for length in range(1, min(len(cur_word), max_length) + 1):
            # Indicates a prefix feature
            prefix = f"PREFIX_{cur_word[:length]}"
            key = ((prefix,), (cur_tag,))
            if key not in self.prefix_tags_dict:
                self.prefix_tags_dict[key] = 0
            self.prefix_tags_dict[key] += 1

    @n_gram(1)
    def get_suffix_tags_pair_count(self, cur_word: str, cur_tag: str) -> None:
        """
        Extract features for the suffix of the current word with a specific tag

        :param cur_word: The current word
        :param cur_tag: The current tag
        """
        max_length = 4
        for length in range(1, min(len(cur_word), max_length) + 1):
            # Indicates a suffix feature
            suffix = f"SUFFIX_{cur_word[-length:]}"
            key = ((suffix,), (cur_tag,))
            if key not in self.suffix_tags_dict:
                self.suffix_tags_dict[key] = 0
            self.suffix_tags_dict[key] += 1

    @n_gram(max_gram)
    def extract_k_grams(self, words, tags) -> None:
        """
        Extract features for the k-grams with their corresponding tags

        :param words: The k-gram words
        :param tags: The corresponding tags
        """
        if (words, tags) not in self.k_gram_dict:
            self.k_gram_dict[(words, tags)] = 0
        self.k_gram_dict[(words, tags)] += 1

    # </editor-fold>
