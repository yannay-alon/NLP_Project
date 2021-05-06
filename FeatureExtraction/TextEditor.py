from typing import Tuple


class TextEditor:
    """
    Reads file with a specific window size \n
    Adds start and end symbols for the line od the text
    """

    def __init__(self, file_path: str, window_size: int):
        self.file_path = file_path
        self.window_size = window_size
        self.text_size = sum(1 for line in open(file_path))  # Number of lined in the text

    def read_file(self, cyclic: bool = False) -> Tuple[str, str]:
        """
        Reads the file and yields its lines \n
        Gets the original lines and the lines with the start and end symbols

        :param cyclic: Whether or not read the file from the beginning after the end
        :return: The original line as it was written in the file <br>
                The decorated line with the extra symbols
        """
        while True:
            with open(self.file_path) as file:
                for line in file:
                    # Remove line breaks (\n) from the end of the line
                    line = line[:-1]

                    # Special symbols for the beginning and ending of the line
                    start = "(╯°□°）╯︵┻━┻".replace(" ", "").replace("_", "")
                    start = f"{start}_{start} " * (self.window_size - 1)

                    end = "┬─┬ノ(゜-゜ノ)".replace(" ", "").replace("_", "")
                    end = f"{end}_{end}"

                    decorated_line = f"{start}{line} {end}"

                    yield line, decorated_line
            if not cyclic:
                break
