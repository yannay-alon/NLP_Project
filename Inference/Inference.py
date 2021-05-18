import numpy as np
import scipy.sparse as sp
from typing import Set, List
from ..FeatureExtraction import History, FeatureID, HistoryHandler
from itertools import combinations


class Inference:

    def __init__(self, feature_id: "FeatureID", weights: np.ndarray, history_handler: "HistoryHandler"):
        self.feature_id = feature_id
        self.weights = weights

        self.tags = history_handler.text_editor.tags
        self.start_symbol = history_handler.text_editor.start
        self.end_symbol = history_handler.text_editor.end

        self.history_length = history_handler.history_length

    def infer(self, words: List[str], beam_size: int = 3):
        prev_probability = {tuple([self.start_symbol] * (self.history_length - 1)): 1}
        cur_probability = dict()
        back_pointers = []
        for history_words in zip(*[words[i:] for i in range(self.history_length)]):
            back_pointers.append(dict())
            for history_closer_tags in combinations(self.tags, self.history_length - 1):
                temp_probabilities = []
                temp_back_pointers = []

                vectors = []
                for opener_tag in self.tags:
                    history = History(words=history_words,
                                      tags=(opener_tag, *history_closer_tags))
                    vectors.append(self.feature_id.history_to_vector(history))
                matrix = sp.vstack(vectors)
                nominators = self.weights @ matrix
                probabilities = nominators / np.sum(nominators)

                for opener_tag, prob in zip(self.tags, probabilities):
                    temp_probabilities.append(prev_probability[(opener_tag, *history_closer_tags[:-1])] * prob)
                    temp_back_pointers.append((opener_tag, *history_closer_tags[:-1]))

                index = np.argmax(temp_probabilities)
                cur_probability[history_closer_tags] = temp_probabilities[index]
                back_pointers[-1][history_closer_tags] = temp_back_pointers[index]

            prev_probability = cur_probability.copy()
            cur_probability.clear()

        predicted_tags = [""] * len(words)
        key = max(prev_probability, key=prev_probability.get)
        predicted_tags[-self.history_length + 1:] = [*key]
        for k in range(len(words) - self.history_length + 1, 0, -1):
            index = k - 1
            key = tuple(predicted_tags[index: index + self.history_length - 1])
            predicted_tags[index] = back_pointers[index + self.history_length - 1][key]

        return predicted_tags
        # prev_probability = {tuple([self.start_symbol] * (self.history_length - 1)): 1}
        # current_probability = dict()
        #
        # # FIXME: Getting the decorated sentences
        # for history_words in zip(*[words[i:] for i in range(self.history_length)]):
        #     beam_probabilities = []
        #     beam_back_pointers = []
        #     for cur_tag in self.tags:
        #         temp_probabilities = []
        #         temp_back_pointers = []
        #         for history_inner_tags in prev_probability.keys():
        #             history = History(words=history_words,
        #                               tags=(*history_inner_tags, cur_tag))
        #             vector = self.feature_id.history_to_vector(history)
        #
        #             temp_probabilities.append(prev_probability[history_inner_tags] * self.weights @ vector)
        #             temp_back_pointers.append(cur_tag)
        #
        #         index = np.argmax(temp_probabilities)
        #         beam_probabilities.append(temp_probabilities[index])
        #         beam_back_pointers.append(temp_back_pointers[index])
        #
        #     indices = np.argpartition(beam_probabilities, beam_size)[:beam_size]
