from collections import OrderedDict


class FeatureID:

    def __init__(self, feature_statistics, threshold):
        self.feature_statistics = feature_statistics  # statistics class, for each featue gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated
        self.n_tag_pairs = 0  # Number of Word\Tag pairs features

        # Init all features dictionaries
        self.words_tags_dict = OrderedDict()

    def get_word_tag_pairs(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                split_words = line.split(' ')
                # del split_words[-1]

                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    if ((cur_word, cur_tag) not in self.words_tags_dict) \
                            and (self.feature_statistics.words_tags_dict[(cur_word, cur_tag)] >= self.threshold):
                        self.words_tags_dict[(cur_word, cur_tag)] = self.n_tag_pairs
                        self.n_tag_pairs += 1
        self.n_total_features += self.n_tag_pairs

    # --- ADD YOURE CODE BELOW --- #
