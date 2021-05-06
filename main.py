from FeatureExtraction import FeatureStatistics, FeatureID
import pandas as pd
from typing import List


def polynomial_threshold(coefficients: List[float]):
    def threshold(length: int):
        total = 0
        for index, coefficient in enumerate(coefficients):
            total += coefficient * length ** index
        return int(total)

    return threshold


def main():
    file_path = r"Data/train1.wtag"
    feature_statistics = FeatureStatistics(file_path)
    feature_id = FeatureID(feature_statistics, [
        polynomial_threshold([25]),  # Capital
        polynomial_threshold([20]),  # Prefix
        polynomial_threshold([15]),  # Suffix
        polynomial_threshold([24, -10, 1]),
    ])
    test_dict = feature_id.features_dict
    pd.DataFrame(list(test_dict.items()), columns=["Key", "Value"]).to_csv("output2.csv", columns=["Key", "Value"])
    print(len(test_dict))


if __name__ == '__main__':
    main()
