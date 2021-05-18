from Project.FeatureExtraction import FeatureStatistics, FeatureID, HistoryHandler, History
from Project.Train import Optimizer
from os import path


def debugging():
    history_to_check = History(('(╯°□°）╯︵┻━┻', '(╯°□°）╯︵┻━┻', 'McDonnell'),
                               ('(╯°□°）╯︵┻━┻', '(╯°□°）╯︵┻━┻', "''"))
    features_file_path = r"features.json"
    feature_id = FeatureID.read_features_from_json(features_file_path)

    print(feature_id.history_to_vector(history_to_check))


def test_infer(history: "History"):
    pass


def main():
    file_path = r"Data/train1.wtag"
    features_file_path = r"features.json"
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

    optimizer = Optimizer(feature_id, history_handler, "weights.pkl")
    optimizer.optimize()


if __name__ == '__main__':
    main()
