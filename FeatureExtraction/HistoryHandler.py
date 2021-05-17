from ..FeatureExtraction.History import History
from typing import Tuple, Iterable
import random
import math


class HistoryHandler:
    """
    Creates the histories from the text file and converts the histories into feature vectors
    """

    def __init__(self, file_path: str, window_size: int):
        self.text_editor = TextEditor(file_path, window_size)
        self.history_length = window_size

    def create_histories(self, max_number: int = None, style: str = "ALL", **kwargs) -> Iterable["History"]:
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

        def increment(start, step, end, cyclic=True):
            # TODO: limit the number of histories to be the number of histories in the text

            yield_counter = 0  # Counter for the number of the yielded histories
            line_index = -1  # Counter for the index of the line
            for line, decorated_line in self.text_editor.read_file(cyclic=cyclic):
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
                    words, tags = zip(*split_list)
                    yield History(words, tags)

                    yield_counter += 1
                    if end is not None and end <= yield_counter:
                        return

        if style == "ALL":
            # return increment(0, 1, min(max_number, self.text_editor.text_size))
            return increment(0, 1, max_number, False)
        elif style == "RANDOM":
            return increment(random.randint(0, self.text_editor.text_size // 2),
                             random.randint(1, int(math.sqrt(self.text_editor.text_size))), max_number)
        elif style == "INCREMENT":
            return increment(kwargs["start"], kwargs["step"], max_number)


class TextEditor:
    """
    Reads file with a specific window size \n
    Adds start and end symbols for the line od the text
    """

    def __init__(self, file_path: str, window_size: int):
        self.file_path = file_path
        self.window_size = window_size
        self.text_size = 0
        self.tags = set()
        self.words = set()

        with open(file_path) as file:
            for line in file:
                self.text_size += 1

                word_tag_list = line.strip("\n").split(" ")
                words, tags = zip(*(word_tag.split("_") for word_tag in word_tag_list))
                self.tags.update(tags)
                self.words.update(words)

    def read_file(self, cyclic: bool = False) -> Tuple[str, str]:
        """
        Reads the file and yields its lines \n
        Gets the original lines and the lines with the start and end symbols

        :param cyclic: Whether or not read the file from the beginning after the end
        :return: The original line as it was written in the file <br>
                The decorated line with the extra symbols
        """
        delimiters = ["_", " "]

        # Special symbols for the beginning and ending of the line
        start = "(╯°□°）╯︵┻━┻"
        end = "┬─┬ノ(゜-゜ノ)"

        # Remove any delimiter (just in case)
        for delimiter in delimiters:
            start = start.replace(delimiter, "")

        for delimiter in delimiters:
            start = start.replace(delimiter, "")

        # Make the special symbols as word_tag pairs
        start = f"{start}_{start} " * (self.window_size - 1)
        end = f"{end}_{end}"

        while True:
            with open(self.file_path) as file:
                for line in file:
                    # Remove line breaks (\n) from the end of the line
                    line = line.strip("\n")

                    decorated_line = f"{start}{line} {end}"
                    yield line, decorated_line

            if not cyclic:
                break
