from collections import OrderedDict
import scipy.sparse
from .FeatureStatistics import FeatureStatistics
from .History import History


class FeatureID:
    """
    Filters the features by thresholds and generates a unique index for each feature
    """

    def __init__(self, feature_statistics: "FeatureStatistics"):
        """
        Create a FeatureID object

        :param feature_statistics: A FeatureStatistics object to use
        """
        self.feature_statistics = feature_statistics

        self.id_counter = 0  # counter for the unique ids

        # Initialize feature dictionary
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
        for key, count in self.feature_statistics.feature_dictionary.items():
            if key.threshold <= count:
                self.features_dict[key] = self.id_counter
                self.id_counter += 1

    def history_to_vector(self, history: "History"):
        feature_vector = scipy.sparse.dok_matrix((self.number_of_features, 1), dtype=int)
        for key in self.feature_statistics.get_keys(history):
            try:
                feature_vector[self.features_dict[key], 0] = 1
            except KeyError:
                pass
        return feature_vector.tocsr()
