from collections import OrderedDict
from .FeatureStatistics import FeatureStatistics


class FeatureID:

    def __init__(self, feature_statistics: "FeatureStatistics", thresholds):
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.thresholds = thresholds  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated
        self.n_tag_pairs = 0  # Number of Word|Tag pairs features

        # Init all features dictionaries
        self.words_tags_dict = OrderedDict()

        # Extract all features dictionaries
        self.get_word_tag_pairs()

    def get_word_tag_pairs(self):
        """
            Extract all relevant word|tag pairs from feature statistics
        """
        for dictionary, threshold in zip(self.feature_statistics.dictionaries, self.thresholds):
            for (cur_word, cur_tag), count in dictionary.items():
                if threshold <= count:
                    self.words_tags_dict[(cur_word, cur_tag)] = self.n_tag_pairs
                    self.n_tag_pairs += 1
        self.n_total_features += self.n_tag_pairs

    # TODO: --- ADD YOUR CODE BELOW --- #
