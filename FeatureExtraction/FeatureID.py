from collections import OrderedDict
from .FeatureStatistics import FeatureStatistics
from typing import List, Callable


class FeatureID:
    """
    Filters the features by thresholds and generates a unique index for each feature
    """

    def __init__(self, feature_statistics: "FeatureStatistics", thresholds: List[Callable[[int], int]]):
        """
        Create a FeatureID object

        :param feature_statistics: A FeatureStatistics object to use
        :param thresholds: A list of functions, <br>
                each function gets the number of grams in the n-gram and returns the threshold
        """
        self.feature_statistics = feature_statistics
        self.thresholds = thresholds

        self.id_counter = 0  # counter for the unique ids

        # Initialize all features dictionaries
        self.features_dict = OrderedDict()

        # Extract all features dictionaries
        self.serialize_features()

    @property
    def number_of_features(self):
        return self.id_counter

    def serialize_features(self):
        """
        Extract all relevant features from feature-statistics
        """
        for dictionary, threshold in zip(self.feature_statistics.dictionaries, self.thresholds):
            for (cur_word, cur_tag), count in dictionary.items():
                if threshold(len(cur_word)) <= count:
                    self.features_dict[(cur_word, cur_tag)] = self.id_counter
                    self.id_counter += 1
