import numpy as np
import scipy.sparse as sp
from typing import Tuple
from ..FeatureExtraction import History, FeatureID, HistoryHandler
from itertools import product


class Inference:

    def __init__(self, feature_id: FeatureID, weights: np.ndarray, history_handler: HistoryHandler):
        self.feature_id = feature_id
        self.weights = weights

        self.tags = history_handler.text_editor.tags
        self.start_symbol = history_handler.text_editor.start
        self.end_symbol = history_handler.text_editor.end

        self.history_length = history_handler.history_length

    def infer(self, words: Tuple[str, ...], beam_size: int = 5):
        words = [self.start_symbol] * (self.history_length - 1) + [*words] + [self.end_symbol]
        sentence_length = len(words)

        normal_tags = list(self.tags.difference([self.start_symbol, self.end_symbol]))

        prev_probability = {tuple([self.start_symbol] * (self.history_length - 1)): 1}
        cur_probability = dict()
        back_pointers = []
        for index, history_words in enumerate(zip(*[words[i:] for i in range(self.history_length)])):
            back_pointers.append(dict())

            start_counters = max(0, self.history_length - 2 - index)
            end_counters = 1 if index == sentence_length - self.history_length else 0
            normal_counter = self.history_length - 1 - start_counters - end_counters
            relevant_closer_tags = [[self.start_symbol]] * start_counters + [normal_tags] * normal_counter + \
                                   [[self.end_symbol]] * end_counters

            for prev_tags, prev_value in prev_probability.items():
                vectors = []
                keys = []
                for closer_tag in relevant_closer_tags[-1]:
                    key = (*prev_tags[1:], closer_tag)
                    keys.append(key)
                    history = History(words=history_words,
                                      tags=(*prev_tags, closer_tag))
                    vector = self.feature_id.history_to_vector(history)
                    vectors.append(vector)
                matrix = sp.hstack(vectors)
                numerators = np.exp(self.weights @ matrix)
                probabilities = numerators / np.sum(numerators) * prev_value
                for key, probability in zip(keys, probabilities):
                    if probability > cur_probability.get(key, 0):
                        cur_probability[key] = probability
                        back_pointers[index][key] = prev_tags[0]

            # for history_closer_tags in product(*relevant_closer_tags):
            #
            #     relevant_opener_tags = [key[0] for key in prev_probability.keys() if
            #                             key[1:] == history_closer_tags[:-1]]
            #     if len(relevant_opener_tags) == 0:
            #         continue
            #
            #     temp_probabilities = []
            #     temp_back_pointers = []
            #
            #     for opener_tag in relevant_opener_tags:
            #         prev_value = prev_probability.get((opener_tag, *history_closer_tags[:-1]), 0)
            #         if prev_value == 0:
            #             continue
            #         vectors = []
            #         current_vector_index = None
            #         for tag_index, closer_tag in enumerate(relevant_closer_tags[-1]):
            #             history = History(words=history_words,
            #                               tags=(opener_tag, *history_closer_tags[:-1], closer_tag))
            #             vector = self.feature_id.history_to_vector(history)
            #             vectors.append(vector)
            #             if history_closer_tags[-1] == closer_tag:
            #                 current_vector_index = tag_index
            #         matrix = sp.hstack(vectors)
            #         numerators = np.exp(self.weights @ matrix)
            #         prob = numerators[current_vector_index] / np.sum(numerators)
            #
            #         temp_probabilities.append(prev_value * prob)
            #         temp_back_pointers.append(opener_tag)
            #
            #     argmax_index = int(np.argmax(temp_probabilities))
            #     cur_probability[history_closer_tags] = temp_probabilities[argmax_index]
            #     back_pointers[-1][history_closer_tags] = temp_back_pointers[argmax_index]

            temp_probability = sorted(cur_probability.items(), key=lambda item: item[1], reverse=True)[:beam_size]
            prev_probability = dict(temp_probability)
            cur_probability.clear()

        predicted_tags = [""] * sentence_length
        key = max(prev_probability, key=prev_probability.get)
        predicted_tags[-self.history_length + 1:] = [*key]
        for k in range(sentence_length - self.history_length + 1, 0, -1):
            key = tuple(predicted_tags[k: k + self.history_length - 1])
            predicted_tags[k - 1] = back_pointers[k - 1][key]

        return predicted_tags[self.history_length - 1: -1]
