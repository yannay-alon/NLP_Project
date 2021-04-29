from .FeatureID import FeatureID
from typing import List


class History:

    def __init__(self, feature_id="FeatureID"):
        self.feature_id = feature_id

    def history_to_vector(self, history_words: List[str], history_tags: List[str]):
        features = []
        pass
