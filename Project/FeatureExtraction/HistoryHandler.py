from ..FeatureExtraction.History import History
from typing import List
import random
import math


class HistoryHandler:
    """
    Creates the histories from the text file and converts the histories into feature vectors
    """

    def __init__(self, file_path: str, window_size: int):
        self.text_editor = TextEditor(file_path, window_size)
        self.history_length = window_size

    def create_histories(self, max_number: int = None, style: str = "ALL", **kwargs) -> List[History]:
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

        def to_histories(lines: List[str]) -> List[History]:
            histories = []
            for line in lines:
                split_words = line.split(" ")

                k_grams = zip(*[split_words[i:] for i in range(self.history_length + 1)])
                for k_gram in k_grams:
                    split_list = (word_tag.split("_") for word_tag in k_gram)
                    words, tags = zip(*split_list)
                    histories.append(History(words[:-1], tags[:-1], tuple([words[-1]])))
            return histories

        decorated_lines = self.text_editor.decorated_lines
        if style == "ALL":
            chosen_lines = decorated_lines
        elif style == "RANDOM":
            chosen_lines = random.sample(decorated_lines, min(max_number, len(decorated_lines)))
        elif style == "INCREMENT":
            start = kwargs["start"]
            step = kwargs["step"]
            indices = range(start, step, start + (max_number - 1) * step)
            chosen_lines = [decorated_lines[i % len(decorated_lines)] for i in indices]
        else:
            chosen_lines = []
        histories = to_histories(chosen_lines)
        return histories


class TextEditor:
    """
    Reads file with a specific window size \n
    Adds start and end symbols for the line od the text
    """

    def __init__(self, file_path: str, window_size: int):
        self.file_path = file_path
        self.window_size = window_size
        self.text_size = 0

        # Special symbols for the beginning and ending of the line
        self.start = "?????????"
        self.end = "?????????"

        self.delimiters = ["_", " "]

        # Remove any delimiter (just in case)
        for delimiter in self.delimiters:
            self.start = self.start.replace(delimiter, "")

        for delimiter in self.delimiters:
            self.end = self.end.replace(delimiter, "")

        self.tags = {self.start, self.end}
        self.words = {self.start, self.end}

        self.decorated_lines = []

        with open(file_path) as file:
            for line in file:
                self.text_size += 1
                line = line.strip("\n")
                word_tag_list = line.split(" ")
                words, tags = zip(*(word_tag.split("_") for word_tag in word_tag_list))
                self.tags.update(tags)
                self.words.update(words)
                self.decorated_lines.append(self.decorate_line(line))

    def decorate_line(self, line: str) -> str:
        # Make the special symbols as word_tag pairs
        start = f"{self.start}_{self.start} " * (self.window_size - 1)
        end = f"{self.end}_{self.end}"
        decorated_line = f"{start}{line} {end}"
        return decorated_line
