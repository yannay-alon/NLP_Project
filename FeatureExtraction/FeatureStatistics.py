from collections import OrderedDict


class FeatureStatistics:

    def __init__(self, file_path: str):
        self.n_total_features = 0  # Total number of features accumulated
        self.file_path = file_path

        # Init all features dictionaries
        self.words_tags_dict = OrderedDict()
        self.capital_tags_dict = OrderedDict()
        self.prefix_tags_dict = OrderedDict()
        self.suffix_tags_dict = OrderedDict()

        self.dictionaries = [self.words_tags_dict,
                             self.capital_tags_dict,
                             self.prefix_tags_dict,
                             self.suffix_tags_dict,
                             ]
        # TODO: ---Add more count dictionaries here---

        # Extract all features dictionaries
        pairs_dictionaries_functions = [self.get_word_tag_pair_count,
                                        self.get_capital_tag_pair_count,
                                        self.get_prefix_tags_pair_count,
                                        self.get_suffix_tags_pair_count,
                                        ]

        self.extract_all_pairs(pairs_dictionaries_functions)

    def extract_all_pairs(self, functions):
        with open(self.file_path) as f:
            for line in f:
                split_words = line.split(" ")
                # del split_words[-1]

                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split("_")
                    for func in functions:
                        func(cur_word, cur_tag)

    def get_word_tag_pair_count(self, cur_word: str, cur_tag: str):
        """
            Extract out of text all word/tag pairs
        """

        if (cur_word, cur_tag) not in self.words_tags_dict:
            self.words_tags_dict[(cur_word, cur_tag)] = 0
        self.words_tags_dict[(cur_word, cur_tag)] += 1

    def get_capital_tag_pair_count(self, cur_word: str, cur_tag: str):
        """
            Extract out of text all word/tag pairs
        """
        capital = cur_word[0].isupper()
        if capital:
            if ("Is_Capital", cur_tag) not in self.capital_tags_dict:
                self.capital_tags_dict[("Is_Capital", cur_tag)] = 0
            self.capital_tags_dict[("Is_Capital", cur_tag)] += 1

    def get_prefix_tags_pair_count(self, cur_word: str, cur_tag: str):
        max_length = 4
        for length in range(1, min(len(cur_word), max_length) + 1):
            prefix = cur_word[-length:]
            if (prefix, cur_tag) not in self.prefix_tags_dict:
                self.prefix_tags_dict[(prefix, cur_tag)] = 0
            self.prefix_tags_dict[(prefix, cur_tag)] += 1

    def get_suffix_tags_pair_count(self, cur_word: str, cur_tag: str):
        max_length = 4
        for length in range(1, min(len(cur_word), max_length) + 1):
            suffix = cur_word[-length:]
            if (suffix, cur_tag) not in self.suffix_tags_dict:
                self.suffix_tags_dict[(suffix, cur_tag)] = 0
            self.suffix_tags_dict[(suffix, cur_tag)] += 1
