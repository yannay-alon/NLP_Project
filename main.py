import FeatureExtraction as fE

def main():
    file_path = r"Data/train1.wtag"
    feature_statistics = fE.FeatureStatistics(file_path)
    feature_id = fE.FeatureID(feature_statistics, 20)
    test_dict = feature_id.words_tags_dict()
    print(test_dict)



if __name__ == '__main__':
    main()
