from collections import OrderedDict


class FeatureStatistics:

    def __init__(self, file_path: str):
        self.n_total_features = 0  # Total number of features accumulated
        self.file_path = file_path

        # Init all features dictionaries
        self.words_tags_count_dict = OrderedDict()
        # TODO: ---Add more count dictionaries here---

    def get_word_tag_pair_count(self):
        """
            Extract out of text all word/tag pairs
        """
        with open(self.file_path) as f:
            for line in f:
                split_words = line.split(" ")
                # del split_words[-1]

                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split("_")
                    if (cur_word, cur_tag) not in self.words_tags_count_dict:
                        self.words_tags_count_dict[(cur_word, cur_tag)] = 0
                    self.words_tags_count_dict[(cur_word, cur_tag)] += 1

    # --- ADD YOUR CODE BELOW --- #


class FeatureID:

    def __init__(self, feature_statistics: "FeatureStatistics", threshold):
        self.feature_statistics = feature_statistics  # statistics class, for each featue gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated
        self.n_tag_pairs = 0  # Number of Word|Tag pairs features

        # Init all features dictionaries
        self.words_tags_dict = OrderedDict()

    def get_word_tag_pairs(self):
        """
            Extract all relevant word|tag pairs from feature statistics
        """
        for (cur_word, cur_tag), count in self.feature_statistics.words_tags_count_dict.items():
            if self.threshold <= count:
                self.words_tags_dict[(cur_word, cur_tag)] = self.n_tag_pairs
                self.n_tag_pairs += 1
        self.n_total_features += self.n_tag_pairs

    # TODO: --- ADD YOUR CODE BELOW --- #
