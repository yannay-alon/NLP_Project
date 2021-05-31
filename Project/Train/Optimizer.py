from ..FeatureExtraction.HistoryHandler import HistoryHandler
from ..FeatureExtraction.FeatureID import FeatureID
from ..FeatureExtraction.History import History
import numpy as np
from typing import Iterable, List, Tuple
from scipy.optimize import fmin_l_bfgs_b as minimize
import scipy.sparse as sp


class Optimizer:
    """
    Optimize the weights vector in order to maximize the likelihood of the data-set
    """

    def __init__(self, feature_id: FeatureID, history_handler: HistoryHandler, path: str):
        self.feature_id = feature_id
        self.history_handler = history_handler
        self.weights = np.empty(feature_id.number_of_features)

        self.path = path
        self.initialize_weight()

    def _preprocess_histories(self, histories: Iterable[History]) -> Tuple[sp.coo_matrix, List[sp.coo_matrix]]:
        """
        Create the vectors of the given histories and the matrices of the histories with changed last tag

        :param histories: The histories to create the feature vectors from
        :return: The vectors of the histories as a coo_matrix (each column is a vector feature)<br>
                The altered vectors for each history:<br>
                (each element in the list is corresponding to a different history, each column is a vector feature)
        """
        vectors = []
        alter_matrices = []
        tags = list(self.history_handler.text_editor.tags)

        histories = list(histories)
        for history in histories:
            vectors.append(self.feature_id.history_to_vector(history))

            new_vectors = []
            for tag in tags:
                # change the last tag and create the altered feature vector
                alternated_history = History(history.words, (*(history.tags[:-1]), tag), history.next_words)
                new_vectors.append(self.feature_id.history_to_vector(alternated_history))
            alter_matrices.append(sp.hstack(new_vectors))

        vectors = sp.hstack(vectors)
        return vectors, alter_matrices

    @staticmethod
    def objective(weights: np.ndarray, vectors: sp.coo_matrix, alter_matrices: List[sp.coo_matrix],
                  regularization: float) -> Tuple[float, np.ndarray]:
        """
        Calculate the objective and the gradient at the given weights vector

        :param weights: The weights vector
        :param vectors: The feature vectors of the histories
        :param alter_matrices: The matrices of the histories with altered last tag
        :param regularization: The regularization coefficient
        :return: The negative likelihood and the negative gradient
        """

        n = len(alter_matrices)

        vf = lambda vec: weights @ vec

        # \sum_{i=1}^{n} { f(x_{i}, y_{i} }
        empirical_counts = np.sum(vectors.A, axis=1)

        # v^T \sum_{i=1}^{n} { f(x_{i}, y_{i} }
        linear_term = float(vf(empirical_counts))

        # at position i, j: e^{v^T f(x_{i}, y'_{j}}
        numerators = [np.exp(vf(alter_matrices[i])) for i in range(n)]

        # at position i: \sum_{y' \in Y} { e^{v^T f(x_{i}, y' }
        denominators = np.array([np.sum(numerators[i]) for i in range(n)])

        # \sum_{i=1}^{n} { log {\sum_{y' \in Y} { e^{v^T f(x_{i}, y' } } }
        normalization_term = np.sum(np.log(denominators))

        # \sum_{i=1}^{n}{\frac{\sum_{y'\in Y}{f(x_{i},y')e^{v^T f(x_{i},y'}} }{\sum_{y'\in Y}{e^{v^T f(x_{i},y')}}}
        expected_counts = np.add.reduce([alter_matrices[i] @ numerators[i].T / denominators[i] for i in range(n)])
        # TODO: check the above line - is it the correct calculation

        # \frac{1}{2} \lambda {\norm v \norm}^2
        regularization_term = 1 / 2 * regularization * np.linalg.norm(weights) ** 2

        # \lambda v
        regularization_gradient = regularization * weights

        likelihood = linear_term - normalization_term - regularization_term
        gradient = empirical_counts - expected_counts - regularization_gradient
        return -likelihood, -gradient

    def optimize(self):
        """
        Optimize the weights vector using (Stochastic) Gradient Descent

        :return: The calculated weights vector
        """
        batch_size = 200  # The number of lines in each batch (about 25 histories per line)
        epsilon = 0  # .001  # The gradient threshold (if the norm of the gradient is less than epsilon, stops)
        # Get random histories for this batch
        for iteration in range(30):
            w_0 = self.weights  # The current weights vector

            histories = self.history_handler.create_histories(batch_size, "RANDOM")

            # Calculates the feature vectors required
            vectors, alter_matrices = self._preprocess_histories(histories)
            print(f"Iteration {iteration}\n"
                  f"\tNumber of histories: {len(histories)}")
            args = (vectors, alter_matrices, 0.5)  # The argument for the objective function

            # Gradient Descent for the current batch
            optimal_params = minimize(func=Optimizer.objective, x0=w_0, args=args,
                                      maxiter=10 * int(np.sqrt(batch_size)))

            weights = optimal_params[0]
            score = optimal_params[1]
            grad = optimal_params[2]["grad"]
            print(f"\tScore: {score}\n"
                  f"\tGradient norm: {np.linalg.norm(grad)}")

            self.weights = weights
            self.save_to_pickle()  # Save the new weights

            if np.linalg.norm(grad) < epsilon:  # Check the stop condition
                break

            batch_size = int(min(batch_size * 1.5, 500))

        return self.weights

    def initialize_weight(self):
        """
        Initialize the weights (from file if exists)
        """
        try:
            # If the file exists, read the weights from the file
            weights = np.load(self.path, allow_pickle=True)
            self.weights = weights
        except IOError:
            # Initialize random weights
            self.weights = np.random.normal(0, 0.5, self.feature_id.number_of_features)

    def save_to_pickle(self):
        """
        Save the current weights as a pickle file
        :return:
        """
        self.weights.dump(self.path)
