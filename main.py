from Project.FeatureExtraction import FeatureStatistics, FeatureID, HistoryHandler, History
from Project.Train import Optimizer
from Project.Inference import Inference
from os import path
import time


def test_infer(inference: Inference):
    accuracy = 0
    counter = 0
    sentence_counter = 0

    start = time.time()
    with open(r"Data/test1.wtag") as file:
        for sentence in file:
            split_list = (word_tag.split("_") for word_tag in sentence.strip("\n").split(" "))
            words, tags = zip(*split_list)

            predicted_tags = inference.infer(words, beam_size=15)
            print(f"real tags: {tags}\n"
                  f"predicted: {predicted_tags}\n")

            for real_tag, predicted_tag in zip(tags, predicted_tags):
                if real_tag == predicted_tag:
                    accuracy += 1
                counter += 1
            sentence_counter += 1
            if sentence_counter > 200:
                break
    end = time.time()
    print(f"Accuracy: {accuracy / counter * 100: .2f}%")
    print(f"Timing:\n"
          f"\tTotal time: {end - start : .3f} sec\n"
          f"\tTime per sentence: {(end - start) / sentence_counter: .3f} sec/snt")


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
