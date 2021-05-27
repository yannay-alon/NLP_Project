from collections import OrderedDict
import scipy.sparse
from ..FeatureExtraction.FeatureStatistics import FeatureStatistics
from ..FeatureExtraction.History import History
from ..FeatureExtraction.Key import Key
import pandas as pd


class FeatureID:
    """
    Filters the features by thresholds and generates a unique index for each feature
    """

    def __init__(self, feature_statistics: "FeatureStatistics" = None):
        """
        Create a FeatureID object

        :param feature_statistics: A FeatureStatistics object to use
        """

        self.id_counter = 0  # counter for the unique ids

        # Initialize feature dictionary
        self.features_dict = OrderedDict()

        if feature_statistics is not None:
            self.feature_statistics = feature_statistics
            # Extract all features dictionaries
            self.serialize_features()
        else:
            self.feature_statistics = FeatureStatistics([])

    @property
    def number_of_features(self):
        return self.id_counter

    def serialize_features(self):
        """
        Extract all relevant features from feature-statistics
        """
        chosen_features = dict()

        for key, count in self.feature_statistics.feature_dictionary.items():
            if key.threshold <= count:
                self.features_dict[key] = self.id_counter
                self.id_counter += 1
                chosen_features[key] = count

        pd.DataFrame.from_dict(data=chosen_features, orient="index") \
            .to_csv("Features_Threshold.csv", header=False)

    def history_to_vector(self, history: "History"):
        feature_vector = scipy.sparse.dok_matrix((self.number_of_features, 1), dtype=int)
        for key in self.feature_statistics.get_keys(history):
            try:
                feature_vector[self.features_dict[key], 0] = 1
            except KeyError:
                pass
        return feature_vector.tocsr()

    # <editor-fold desc="I/O json">
    @staticmethod
    def read_features_from_json(path: str):

        feature_id = FeatureID()
        feature_id.features_dict = OrderedDict()

        dictionary = pd.read_json(path).to_dict(orient="records")
        for d in dictionary:
            key = Key(tuple(d["Words"]), tuple(d["Tags"]), 1)
            feature_id.features_dict[key] = d["Value"]
        feature_id.id_counter = len(feature_id.features_dict.keys())

        return feature_id

    def save_feature_as_json(self, path: str):
        data = [[[*key.words], [*key.tags], value] for key, value in self.features_dict.items()]
        pd.DataFrame(data, columns=["Words", "Tags", "Value"]) \
            .to_json(path, orient="records")
    # </editor-fold>
