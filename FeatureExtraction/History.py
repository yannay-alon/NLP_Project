from .FeatureID import FeatureID
from .TextEditor import TextEditor
from typing import List, Tuple, Iterable
import random
import math


class History:

    def __init__(self, feature_id: "FeatureID", text_editor: "TextEditor"):
        self.feature_id = feature_id
        self.text_editor = text_editor
        self.history_length = text_editor.window_size

    def history_to_vector(self, history_words: List[str], history_tags: List[str]):
        features = []
        pass

    def create_histories(self, max_number: int = None, style: str = None, **kwargs) -> \
            Iterable[Tuple[Iterable[str], Iterable[str]]]:

        def increment(start, step, end):
            yield_counter = 0
            line_index = -1
            for line, decorated_line in self.text_editor.read_file(cyclic=True):
                line_index += 1

                if line_index < start:
                    continue
                if (line_index - start) % step != 0:
                    continue

                split_words = decorated_line.split(" ")

                k_grams = zip(*[split_words[i:] for i in range(self.history_length)])
                for k_gram in k_grams:
                    split_list = (word_tag.split("_") for word_tag in k_gram)
                    words, tags = list(zip(*split_list))
                    yield words, tags

                    yield_counter += 1
                    if end is not None and end <= yield_counter:
                        raise StopIteration

        if style == "ALL":
            return increment(0, 1, max_number)
        elif style == "RANDOM":
            return increment(random.randint(0, self.text_editor.text_size // 2),
                             random.randint(1, int(math.sqrt(self.text_editor.text_size))),
                             max_number)
        elif style == "INCREMENT":
            return increment(kwargs["start"], kwargs["step"], max_number)
