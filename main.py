from FeatureExtraction import FeatureStatistics, FeatureID, TextEditor
import pandas as pd
from typing import List


def polynomial_threshold(coefficients: List[float]):
    """
    Creates a threshold as a function of the number of elements in the k-gram

    :param coefficients: The coefficients for the polynomial function
    :return: A function that takes the number of elements in the k-gram and returns the threshold
    """

    def threshold(length: int):
        total = 0
        for index, coefficient in enumerate(coefficients):
            total += coefficient * length ** index
        return int(total)

    return threshold


def main():
    file_path = r"Data/train1.wtag"
    max_gram = 3

    text_editor = TextEditor(file_path, max_gram)
    feature_statistics = FeatureStatistics(text_editor)

    feature_id = FeatureID(feature_statistics, [
        polynomial_threshold([25]),  # Capital
        polynomial_threshold([20]),  # Prefix
        polynomial_threshold([15]),  # Suffix
        polynomial_threshold([24, -10, 1]),  # n-gram
    ])

    test_dict = feature_id.features_dict

    # Save the features in as csv file
    pd.DataFrame(list(test_dict.items()), columns=["Key", "Value"]). \
        to_csv("features.csv", columns=["Key", "Value"], index=False)
    print(len(test_dict))


if __name__ == '__main__':
    main()
