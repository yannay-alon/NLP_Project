from typing import Tuple


class History:
    def __init__(self, words: Tuple[str, ...], tags: Tuple[str, ...]):
        self.words = words
        self.tags = tags

    def __len__(self):
        return len(self.words)

    def __eq__(self, other):
        if isinstance(other, History):
            return self.words == other.words and self.tags == other.tags
        return False

    def __hash__(self):
        return hash(self.words) ^ hash(self.tags)

    def __repr__(self):
        return f"words: {self.words}, tags: {self.tags}"

    def __str__(self):
        return f"words: {self.words}, tags: {self.tags}"
