import time
from scipy.optimize import fmin_l_bfgs_b as minimizer
import scipy.sparse as sp
from Project.FeatureExtraction.HistoryHandler import HistoryHandler
from Project.FeatureExtraction.FeatureID import FeatureID
import numpy as np


class Optimizer:

    def __init__(self, feature_id: "FeatureID", history_handler: "HistoryHandler"):
        self.feature_id = feature_id
        self.history_handler = history_handler
        self.weights = sp.random(feature_id.number_of_features, 1, density=1)

    @staticmethod
    def objective(weights: sp.spmatrix, vectors: sp.coo_matrix, regularization: float):
        linear_term = weights.transpose().dot(vectors.dot(np.ones(vectors.get_shape()[1])))
        # normalized_term =

    def optimize(self):
        pass
