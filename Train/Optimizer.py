from Project.FeatureExtraction.HistoryHandler import HistoryHandler
from Project.FeatureExtraction.FeatureID import FeatureID
from ..FeatureExtraction.History import History
import numpy as np
from typing import Iterable
from scipy.optimize import fmin_l_bfgs_b as minimizer
import scipy.sparse as sp


class Optimizer:

    def __init__(self, feature_id: "FeatureID", history_handler: "HistoryHandler"):
        self.feature_id = feature_id
        self.history_handler = history_handler
        self.weights = sp.random(feature_id.number_of_features, 1, density=1)

    def objective(self, weights: sp.spmatrix, histories: Iterable["History"], regularization: float):
        vectors = []
        alter_vectors = []
        tags = self.history_handler.text_editor.tags
        for history in histories:
            vectors.append(self.feature_id.history_to_vector(history))

            new_vectors = []
            for tag in tags:
                alternated_history = History(history.words, (*(history.tags[:-1]), tag))
                new_vectors.append(self.feature_id.history_to_vector(alternated_history))
            alter_vectors.append(sp.hstack(new_vectors))

        vectors = sp.hstack(vectors)
        n = len(alter_vectors)

        vf = lambda vec: weights.transpose().dot(vec)

        linear_term = vf(vectors.dot(np.ones(n)))

        denominators = np.array([np.sum(np.exp(vf(alter_vectors[i]).toarray())) for i in range(n)])
        normalized_term = np.sum(np.log(denominators))
        regularization_term = 1 / 2 * regularization * np.sum(np.abs(weights))  # norm 1

        likelihood = linear_term - normalized_term - regularization_term

    def optimize(self):
        pass
