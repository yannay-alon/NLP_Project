from ..FeatureExtraction.HistoryHandler import HistoryHandler
from ..FeatureExtraction.FeatureID import FeatureID
from ..FeatureExtraction.History import History
import numpy as np
from typing import Iterable
from scipy.optimize import fmin_l_bfgs_b as minimize
import scipy.sparse as sp


class Optimizer:

    def __init__(self, feature_id: "FeatureID", history_handler: "HistoryHandler", path: str):
        self.feature_id = feature_id
        self.history_handler = history_handler
        self.weights = np.empty(feature_id.number_of_features)

        self.path = path
        self.initialize_weight()

    def objective(self, weights: np.ndarray, histories: Iterable["History"], regularization: float):
        vectors = []
        alter_matrices = []
        tags = self.history_handler.text_editor.tags

        for history in histories:
            vectors.append(self.feature_id.history_to_vector(history))

            new_vectors = []
            for tag in tags:
                alternated_history = History(history.words, (*(history.tags[:-1]), tag))
                new_vectors.append(self.feature_id.history_to_vector(alternated_history))
            alter_matrices.append(sp.hstack(new_vectors))

        vectors = sp.hstack(vectors)
        n = len(alter_matrices)

        vf = lambda vec: vec.transpose().dot(weights)

        empirical_counts = vectors.dot(np.ones(n))
        linear_term = float(vf(empirical_counts))

        numerators = [np.exp(vf(alter_matrices[i])) for i in range(n)]
        denominators = np.array([np.sum(numerators[i]) for i in range(n)])
        normalized_term = np.sum(np.log(denominators))
        expected_counts = np.add.reduce([alter_matrices[i] @ numerators[i].T / denominators[i] for i in range(n)])

        regularization_term = 1 / 2 * regularization * np.linalg.norm(weights) ** 2
        regularization_gradient = regularization * weights

        likelihood = linear_term - normalized_term - regularization_term

        score = empirical_counts - expected_counts - regularization_gradient

        return -likelihood, -score

    def optimize(self):
        for iteration in range(10):
            args = (list(self.history_handler.create_histories(100, "RANDOM")), 2)
            w_0 = self.weights

            optimal_params = minimize(func=self.objective, x0=w_0, args=args, maxiter=10)

            weights = optimal_params[0]
            final_score = optimal_params[1]
            grad = optimal_params[2]["grad"]
            print(f"{final_score}")
            print(f"norm: {np.linalg.norm(grad)}")

            self.weights = weights
        self.save_to_pickle()

        return self.weights

    def initialize_weight(self):
        try:
            weights = np.load(self.path, allow_pickle=True)
            self.weights = weights
        except IOError:
            self.weights = np.random.normal(0, 0.5, self.feature_id.number_of_features)

    def save_to_pickle(self):
        self.weights.dump(self.path)
