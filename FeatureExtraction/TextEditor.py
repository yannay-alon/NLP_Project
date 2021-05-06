from typing import Tuple


class TextEditor:

    def __init__(self, file_path: str, window_size: int):
        self.file_path = file_path
        self.window_size = window_size
        self.text_size = sum(1 for line in open(file_path))

    def read_file(self, cyclic: bool = False) -> Tuple[str, str]:
        while True:
            with open(self.file_path) as file:
                for line in file:
                    # Remove \n at the end of the line
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
