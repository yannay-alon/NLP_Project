from FeatureExtraction import FeatureStatistics, FeatureID
import pandas as pd


def main():
    file_path = r"Data/train1.wtag"
    feature_statistics = FeatureStatistics(file_path)
    feature_id = FeatureID(feature_statistics, [20 for d in feature_statistics.dictionaries])
    test_dict = feature_id.words_tags_dict
    pd.DataFrame(list(test_dict.items()), columns=["Key", "Value"]).to_csv("output2.csv", columns=["Key", "Value"])
    print(len(test_dict))


if __name__ == '__main__':
    main()
