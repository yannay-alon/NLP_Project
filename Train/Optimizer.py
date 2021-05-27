from ..FeatureExtraction.HistoryHandler import HistoryHandler
from ..FeatureExtraction.FeatureID import FeatureID
from ..FeatureExtraction.History import History
import numpy as np
from typing import Iterable, List
from scipy.optimize import fmin_l_bfgs_b as minimize
import scipy.sparse as sp


class Optimizer:

    def __init__(self, feature_id: "FeatureID", history_handler: "HistoryHandler", path: str):
        self.feature_id = feature_id
        self.history_handler = history_handler
        self.weights = np.empty(feature_id.number_of_features)

        self.path = path
        self.initialize_weight()

    def _preprocess_histories(self, histories: Iterable["History"]):
        vectors = []
        alter_matrices = []
        tags = list(self.history_handler.text_editor.tags)

        histories = list(histories)
        for history in histories:

            vectors.append(self.feature_id.history_to_vector(history))

            new_vectors = []
            for tag in tags:
                alternated_history = History(history.words, (*(history.tags[:-1]), tag))
                new_vectors.append(self.feature_id.history_to_vector(alternated_history))
            alter_matrices.append(sp.hstack(new_vectors))

        vectors = sp.hstack(vectors)
        return vectors, alter_matrices

    @staticmethod
    def objective(weights: np.ndarray, vectors: sp.coo_matrix, alter_matrices: List[sp.coo_matrix],
                  regularization: float):

        n = len(alter_matrices)

        vf = lambda vec: weights @ vec

        empirical_counts = np.sum(vectors.A, axis=1)
        linear_term = float(vf(empirical_counts))

        numerators = [np.exp(vf(alter_matrices[i])) for i in range(n)]
        denominators = np.array([np.sum(numerators[i]) for i in range(n)])
        normalization_term = np.sum(np.log(denominators))
        expected_counts = np.add.reduce([alter_matrices[i] @ numerators[i].T / denominators[i] for i in range(n)])

        regularization_term = 1 / 2 * regularization * np.linalg.norm(weights) ** 2
        regularization_gradient = regularization * weights

        likelihood = linear_term - normalization_term - regularization_term
        gradient = empirical_counts - expected_counts - regularization_gradient
        return -likelihood, -gradient

    def optimize(self):
        batch_size = 10
        for iteration in range(50):
            w_0 = self.weights

            histories = self.history_handler.create_histories(batch_size, "RANDOM")
            vectors, alter_matrices = self._preprocess_histories(histories)
            args = (vectors, alter_matrices, 2)

            optimal_params = minimize(func=Optimizer.objective, x0=w_0, args=args,
                                      maxiter=10 * int(np.sqrt(batch_size)))

            weights = optimal_params[0]
            score = optimal_params[1]
            grad = optimal_params[2]["grad"]
            print(f"iteration: {iteration}\n"
                  f"\tScore: {score}\n"
                  f"\tGradient norm: {np.linalg.norm(grad)}")

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
