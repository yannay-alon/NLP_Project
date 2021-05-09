from typing import Tuple, Union


class Key:
    def __init__(self, words: Tuple[Union[str]], tags: Tuple[Union[str]]):
        self.words = words
        self.tags = tags
