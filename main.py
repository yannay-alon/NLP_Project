from FeatureExtraction import FeatureStatistics, FeatureID, HistoryHandler
import pandas as pd


def main():
    file_path = r"Data/test1.wtag"
    max_gram = 3
    # <editor-fold desc="create feature dict">

    history_handler = HistoryHandler(file_path, max_gram)
    feature_statistics = FeatureStatistics(history_handler.create_histories(1000, "ALL"))
    feature_dict = feature_statistics.feature_dictionary
    feature_id = FeatureID(feature_statistics)

    test_dict = feature_id.features_dict

    # Save the features in as csv file
    pd.DataFrame(list(test_dict.items()), columns=["Key", "Value"]). \
        to_csv("features.csv", columns=["Key", "Value"], index=False)
    print(len(test_dict))

    # </editor-fold>

    for history in history_handler.create_histories(1, "RANDOM"):
        vector = feature_id.history_to_vector(history)
        print(vector)
        break


if __name__ == '__main__':
    main()
