from typing import Tuple, Union


class Key:
    def __init__(self, words: Tuple[Union[str]], tags: Tuple[Union[str]], threshold: int):
        self.words = words
        self.tags = tags
        self.threshold = threshold

    def __len__(self):
        return max(len(self.words), len(self.tags))

    def __eq__(self, other):
        if isinstance(other, Key):
            return self.words == other.words and self.tags == other.tags
        return False

    def __hash__(self):
        return hash(self.words) ^ hash(self.tags)

    def __repr__(self):
        return f"words: {self.words}, tags: {self.tags}, threshold: {self.threshold}"

    def __str__(self):
        return f"words: {self.words}, tags: {self.tags}"
