from typing import Tuple, Union


class History:
    def __init__(self, words: Tuple[str, ...], tags: Tuple[str, ...]):
        self.words = words
        self.tags = tags

    def __len__(self):
        return len(self.words)
