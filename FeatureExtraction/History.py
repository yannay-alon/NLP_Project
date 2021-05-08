from .FeatureID import FeatureID
from .TextEditor import TextEditor
from typing import List, Tuple, Iterable
import random
import math


class History:
    """
    Creates the histories from the text file and converts the histories into feature vectors
    """

    def __init__(self, feature_id: "FeatureID", text_editor: "TextEditor"):
        self.feature_id = feature_id
        self.text_editor = text_editor
        self.history_length = text_editor.window_size

    def history_to_vector(self, history_words: List[str], history_tags: List[str]):
        features = []
        pass

    def create_histories(self, max_number: int = None, style: str = "ALL", **kwargs) -> \
            Iterable[Tuple[Iterable[str], Iterable[str]]]:
        """
        Create the histories from the text editor

        :param max_number: The number of histories to get, if None then reads until the end
        :param style: The style to extract the histories <br>
                    &emsp - ALL: Get all of the histories (or until reaching max_number) <br>
                    &emsp - INCREMENT: Get histories starting at the given start line with the given step size <br>
                    &emsp - RANDOM: Get histories starting at a random line with random steps
        :param kwargs: In case of using INCREMENT: <br>
                        &emsp start - the index of the first line to read <br>
                        &emsp step - the number of lines between each read <br>
                        In any other case, ignores the kwargs
        :return: yields the histories from the read lines
        """

        def increment(start, step, end):
            # TODO: limit the number of histories to be the number of histories in the text

            yield_counter = 0  # Counter for the number of the yielded histories
            line_index = -1  # Counter for the index of the line
            for line, decorated_line in self.text_editor.read_file(cyclic=True):
                line_index += 1

                # traces the index for the increment
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
            # return increment(0, 1, min(max_number, self.text_editor.text_size))
            return increment(0, 1, max_number)
        elif style == "RANDOM":
            return increment(random.randint(0, self.text_editor.text_size // 2),
                             random.randint(1, int(math.sqrt(self.text_editor.text_size))),
                             max_number)
        elif style == "INCREMENT":
            return increment(kwargs["start"], kwargs["step"], max_number)
