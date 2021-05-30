from typing import Tuple


class History:
    def __init__(self, words: Tuple[str, ...], tags: Tuple[str, ...], next_words: Tuple[str, ...] = tuple()):
        self.words = words
        self.tags = tags
        self.next_words = next_words

    def __len__(self):
        return len(self.words)

    def __eq__(self, other):
        if isinstance(other, History):
            return self.words == other.words and self.tags == other.tags and self.next_words == other.next_words
        return False

    def __hash__(self):
        return hash(self.words) ^ hash(self.tags) ^ hash(self.next_words)

    def __repr__(self):
        return f"words: {self.words}, tags: {self.tags}, next words: {self.next_words}"

    def __str__(self):
        return f"words: {self.words}, tags: {self.tags}, next words: {self.next_words}"
