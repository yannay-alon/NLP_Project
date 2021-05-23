from Project.FeatureExtraction import FeatureStatistics, FeatureID, HistoryHandler, History
from Project.Train import Optimizer
from Project.Inference import Inference
from os import path


def debugging():
    history_to_check = History(('(╯°□°）╯︵┻━┻', '(╯°□°）╯︵┻━┻', 'McDonnell'),
                               ('(╯°□°）╯︵┻━┻', '(╯°□°）╯︵┻━┻', "''"))
    features_file_path = r"features.json"
    feature_id = FeatureID.read_features_from_json(features_file_path)

    print(feature_id.history_to_vector(history_to_check))


def test_infer(inference: Inference):
    sentence = "No_DT one_NN can_MD say_VB ._."

    split_list = (word_tag.split("_") for word_tag in sentence.split(" "))
    words, tags = zip(*split_list)

    predicted_tags = inference.infer(words)
    print(f"real tags: {tags}\n"
          f"predicted: {predicted_tags}")

    accuracy = 0
    for real_tag, predicted_tag in zip(tags, predicted_tags):
        if real_tag == predicted_tag:
            accuracy += 1
    print(f"Accuracy: {accuracy / len(tags) * 100: .2f}%")


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
    # optimizer.optimize()

    inference = Inference(feature_id, optimizer.weights, history_handler)
    test_infer(inference)


if __name__ == '__main__':
    main()
