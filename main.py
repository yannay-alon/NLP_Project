from Project.FeatureExtraction import FeatureStatistics, FeatureID, HistoryHandler
from Project.Train import Optimizer
from os import path
import numpy as np
from scipy import sparse


def main():
    file_path = r"Data/train1.wtag"
    features_file_path = r"features.csv"
    max_gram = 3

    # <editor-fold desc="create feature dict">

    history_handler = HistoryHandler(file_path, max_gram)
    if path.exists(features_file_path):
        feature_id = FeatureID.read_features_from_json(features_file_path)
    else:
        feature_statistics = FeatureStatistics(history_handler.create_histories(None, "ALL"))
        feature_id = FeatureID(feature_statistics)

        # Save the features in as csv file
        feature_id.save_feature_as_json(features_file_path)

    # </editor-fold>

    optimizer = Optimizer(feature_id, history_handler)
    optimizer.objective(optimizer.weights, history_handler.create_histories(100, "RANDOM"), 2)


if __name__ == '__main__':
    main()
