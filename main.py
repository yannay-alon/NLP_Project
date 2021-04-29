from FeatureExtraction import FeatureID, FeatureStatistics


def main():
    file_path = r"Data/train1.wtag"
    feature_statistics = FeatureStatistics(file_path)
    feature_id = FeatureID(feature_statistics, [15 for d in feature_statistics.dictionaries])
    test_dict = feature_id.words_tags_dict
    print(len(test_dict))


if __name__ == '__main__':
    main()
