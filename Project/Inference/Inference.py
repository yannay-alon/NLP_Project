import numpy as np
import scipy.sparse as sp
from typing import Tuple, List
from ..FeatureExtraction import History, FeatureID, HistoryHandler


class Inference:
    """
    Responsible to infer the tags of a sentence after training
    """

    def __init__(self, feature_id: FeatureID, weights: np.ndarray, history_handler: HistoryHandler):
        self.feature_id = feature_id
        self.weights = weights

        self.tags = history_handler.text_editor.tags
        self.start_symbol = history_handler.text_editor.start
        self.end_symbol = history_handler.text_editor.end

        self.history_length = history_handler.history_length

    def infer(self, words: Tuple[str, ...], beam_size: int = 5) -> List[str]:
        """
        Infer the tags of the given words using Viterbi (beam search) algorithm

        :param words: The words in the sentence
        :param beam_size: The width of the beam
        :return: The predicted tags as a list
        """
        # Add padding symbols
        words = [self.start_symbol] * (self.history_length - 1) + [*words] + [self.end_symbol] * 2
        sentence_length = len(words) - 1

        normal_tags = list(self.tags.difference([self.start_symbol, self.end_symbol]))

        # Initialize the probability for the base case
        prev_probability = {tuple([self.start_symbol] * (self.history_length - 1)): 1}
        cur_probability = dict()
        back_pointers = []

        # Iterate over the histories in the sentence
        for index, all_words in enumerate(zip(*[words[i:] for i in range(self.history_length + 1)])):
            history_words = all_words[:-1]
            next_word = all_words[-1]
            back_pointers.append(dict())

            # Check if this is the end of the sentence
            if index < sentence_length - self.history_length:
                relevant_closer_tags = normal_tags
            else:
                relevant_closer_tags = [self.end_symbol]

            # Iterate over previous non-zero options
            for prev_tags, prev_value in prev_probability.items():
                vectors = []
                keys = []
                # Create all possible tags for the current word
                for closer_tag in relevant_closer_tags:
                    key = (*prev_tags[1:], closer_tag)
                    keys.append(key)
                    history = History(words=history_words,
                                      tags=(*prev_tags, closer_tag),
                                      next_words=tuple([next_word]))
                    vector = self.feature_id.history_to_vector(history)
                    vectors.append(vector)

                # Calculate the probability of each tag
                matrix = sp.hstack(vectors)
                numerators = np.exp(self.weights @ matrix)
                probabilities = numerators / np.sum(numerators) * prev_value

                # Choose the option with the maximal probability
                for key, probability in zip(keys, probabilities):
                    if probability > cur_probability.get(key, 0):
                        cur_probability[key] = probability
                        back_pointers[index][key] = prev_tags[0]

            # Reduce the options in respect to the beam size
            temp_probability = sorted(cur_probability.items(), key=lambda item: item[1], reverse=True)
            prev_probability = dict(temp_probability[:beam_size])
            cur_probability.clear()

        # Use the back-pointers to find the tags
        predicted_tags = [""] * sentence_length
        key = max(prev_probability, key=prev_probability.get)
        predicted_tags[-self.history_length + 1:] = [*key]
        for k in range(sentence_length - self.history_length + 1, 0, -1):
            key = tuple(predicted_tags[k: k + self.history_length - 1])
            predicted_tags[k - 1] = back_pointers[k - 1][key]

        # Return the tags without the start and end padding
        return predicted_tags[self.history_length - 1: -1]
