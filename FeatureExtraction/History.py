from .FeatureID import FeatureID
from typing import List


class History:

    def __init__(self, feature_id: "FeatureID", history_length: int):
        self.feature_id = feature_id
        self.history_length = history_length

    def history_to_vector(self, history_words: List[str], history_tags: List[str]):
        features = []
        pass

    def create_histories(self, file_path, max_number: int, style: str, **kwargs) -> List["History"]:

        with open(file_path) as f:
            for line in f:
                pass

