from Project.FeatureExtraction import FeatureStatistics, FeatureID, HistoryHandler
from Project.Train import Optimizer
from Project.Inference import Inference
from os import path
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import time


def plot_confusion_matrix(matrix: pd.DataFrame):
    matrix.to_csv("confusion_matrix.csv", header=False)
    matrix_copy = matrix.copy()
    np.fill_diagonal(matrix_copy.values, 0)
    indices = pd.DataFrame(data=matrix_copy, index=matrix.index, columns=matrix.columns).sum().nlargest(10).index
    sn.heatmap(matrix[indices], annot=True)
    plt.show()


def test_infer(inference: Inference):
    accuracy = 0
    counter = 0
    sentence_counter = 0
    max_sentences = 1000

    tags = inference.tags.difference([inference.start_symbol, inference.start_symbol])
    matrix = pd.DataFrame(data=0, index=tags, columns=tags)

    start = time.time()
    with open(r"Data/test1.wtag") as file:
        for sentence in file:
            split_list = (word_tag.split("_") for word_tag in sentence.strip("\n").split(" "))
            words, tags = zip(*split_list)

            predicted_tags = inference.infer(words, beam_size=5)
            # print(f"real tags: {tags}\n"
            #       f"predicted: {predicted_tags}\n")

            for real_tag, predicted_tag in zip(tags, predicted_tags):
                matrix[real_tag][predicted_tag] += 1
                if real_tag == predicted_tag:
                    accuracy += 1
                counter += 1
            sentence_counter += 1
            if (10 * sentence_counter) % max_sentences == 0:
                print(f"Finished {sentence_counter / max_sentences * 100}% of the predictions")
                print(f"Accuracy: {accuracy / counter * 100: .2f}%")
            if max_sentences <= sentence_counter:
                break
    end = time.time()
    print(f"Timing:\n"
          f"\tTotal time: {end - start : .3f} sec\n"
          f"\tTime per sentence: {(end - start) / sentence_counter: .3f} sec/snt")
    plot_confusion_matrix(matrix)


def main():
    file_path = r"Data/train1.wtag"
    features_file_path = r"features.json"

    max_gram = 3
    optimize = True

    # <editor-fold desc="Initialize the features">

    history_handler = HistoryHandler(file_path, max_gram)
    if path.exists(features_file_path):
        feature_id = FeatureID.read_features_from_json(features_file_path)
    else:
        feature_statistics = FeatureStatistics(history_handler.create_histories(None, "ALL"))
        feature_id = FeatureID(feature_statistics)

        # Save the features as csv file
        feature_id.save_feature_as_json(features_file_path)

    # </editor-fold>

    # <editor-fold desc="Optimization">
    optimizer = Optimizer(feature_id, history_handler, "weights.pkl")
    if optimize:
        optimizer.optimize()

    # </editor-fold>

    # <editor-fold desc="Inference">
    inference = Inference(feature_id, optimizer.weights, history_handler)
    test_infer(inference)

    # </editor-fold>


if __name__ == '__main__':
    main()
