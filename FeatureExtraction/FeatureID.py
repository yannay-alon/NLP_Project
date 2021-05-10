from collections import OrderedDict
import scipy.sparse
from Project.FeatureExtraction.FeatureStatistics import FeatureStatistics
from Project.FeatureExtraction.History import History
from Project.FeatureExtraction.Key import Key
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

    # <editor-fold desc="I/O csv">
    @staticmethod
    def read_features_from_csv(path: str):
        def convert_to_tuple(string: str):
            sub_strings = string[1:-1].replace(" ", "").replace("\'", "").split(",")
            if not sub_strings[-1]:
                sub_strings = sub_strings[:-1]
            return tuple(sub_strings)

        feature_id = FeatureID()
        feature_id.features_dict = OrderedDict([(Key(convert_to_tuple(d["Words"]),
                                                     convert_to_tuple(d["Tags"]), 1),
                                                 int(d["Value"])) for d in
                                                pd.read_csv(path).to_dict(into=dict, orient="records")])

        feature_id.id_counter = len(feature_id.features_dict.keys())
        return feature_id

    def save_feature_as_csv(self, path: str):
        data = [(key.words, key.tags, value) for key, value in self.features_dict.items()]
        pd.DataFrame(data, columns=["Words", "Tags", "Value"]) \
            .to_csv(path, columns=["Words", "Tags", "Value"], index=False)
    # </editor-fold>
