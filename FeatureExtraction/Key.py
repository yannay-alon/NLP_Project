from typing import Tuple, Union


class Key:
    def __init__(self, words: Tuple[Union[str]], tags: Tuple[Union[str]], threshold: int):
        self.words = words
        self.tags = tags
        self.threshold = threshold

    def __len__(self):
        return len(self.words)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Key):
            return self.words == other.words and self.tags == other.tags and self.threshold == other.threshold
        return False

    def __hash__(self):
        return hash(self.words) ^ hash(self.tags) ^ hash(self.threshold)

    def __repr__(self):
        return f"words: {self.words}, tags: {self.tags}, threashold: {self.threshold}"

    def __str__(self):
        return f"words: {self.words}, tags: {self.tags}"
