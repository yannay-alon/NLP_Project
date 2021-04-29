from collections import OrderedDict


class FeatureStatistics:

    def __init__(self, file_path: str):
        self.n_total_features = 0  # Total number of features accumulated
        self.file_path = file_path

        # Init all features dictionaries
        self.words_tags_count_dict = OrderedDict()
        # ---Add more count dictionaries here---

    def get_word_tag_pair_count(self):
        """
            Extract out of text all word/tag pairs
        """
        with open(self.file_path) as f:
            for line in f:
                split_words = line.split(" ")
                # del split_words[-1]
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split("_")
                    if (cur_word, cur_tag) not in self.words_tags_count_dict:
                        self.words_tags_count_dict[(cur_word, cur_tag)] = 0
                    self.words_tags_count_dict[(cur_word, cur_tag)] += 1

    # --- ADD YOUR CODE BELOW --- #
