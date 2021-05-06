from collections import OrderedDict
from .FeatureStatistics import FeatureStatistics
from typing import List, Callable


class FeatureID:

    def __init__(self, feature_statistics: "FeatureStatistics", thresholds: List[Callable[[int], int]]):
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.thresholds = thresholds  # feature count threshold - empirical count must be higher than this

        self.id_counter = 0  # Number of Word|Tag pairs features

        # Init all features dictionaries
        self.features_dict = OrderedDict()

        # Extract all features dictionaries
        self.serialize_features()

    @property
    def total_features(self):
        return self.id_counter

    def serialize_features(self):
        """
            Extract all relevant word|tag pairs from feature statistics
        """
        for dictionary, threshold in zip(self.feature_statistics.dictionaries, self.thresholds):
            for (cur_word, cur_tag), count in dictionary.items():
                if threshold(len(cur_word)) <= count:
                    self.features_dict[(cur_word, cur_tag)] = self.id_counter
                    self.id_counter += 1
