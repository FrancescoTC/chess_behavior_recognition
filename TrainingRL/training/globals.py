import os
import inspect as inspect_

from pathlib import Path
from inspect import getsourcefile

def abspath():
    """
    Returns the absolute path of the directory containing the caller module.

    :return: The absolute path of the directory containing the caller module.
    :rtype: str

    # https://stackoverflow.com/questions/16771894/python-nameerror-global-name-file-is-not-defined
    # https://docs.python.org/3/library/inspect.html#inspect.FrameInfo
    # return os.path.dirname(inspect.stack()[1][1]) # type: ignore
    # return os.path.dirname(getsourcefile(lambda:0)) # type: ignore
    """
    return Path(os.path.dirname(getsourcefile(inspect_.stack()[1][0])))  # type: ignore


def setup_path(path: str) -> Path:
    """
    Create path if it does not exist.

    :param path: The path to be created.
    :type path: str
    :return: The created path.
    :rtype: Path
    """
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


class Globals:
    """
    Global variables for the project.
    - MAX_MOVES (int): Due to time constraints, max number of moves per game.
    - REPO_DIR (Path): Repository root directory.
    - DATA_ROOT (Path): Directory for data storage.
    - RUN_DIR (Path): Directory for run logs.
    - LOG_DIR (Path): Directory for general logs.
    - STOCKFISH_PATH (str): Path to the Stockfish binary.
    """

    MAX_MOVES = 89  # Due to time constraints, max number of moves per game

    REPO_DIR = setup_path(Path(abspath()) / "..")
    DATA_ROOT = setup_path(os.getenv("DATA_ROOT") or REPO_DIR)
    RUN_DIR = LOG_DIR = Path()
    STOCKFISH_PATH = os.path.join(DATA_ROOT, "stockfish_2/stockfish/stockfish-ubuntu-x86-64-avx2")

    def print():
        """
        Print the global variables.
        """
        attributes = "\n".join(
            f"{key}={value!r}"
            for key, value in Globals.__dict__.items()
            if not key.startswith("_") and key != "print"
        )
        print(f"Globals({attributes})")

    @staticmethod
    def initialize_tokenizer():
        """
        Initialize the tokenizer metadata.
        """
        base_meta = {
            "vocab_size": 32,
            "itos": {
                0: " ",
                1: "#",
                2: "+",
                3: "-",
                4: ".",
                5: "0",
                6: "1",
                7: "2",
                8: "3",
                9: "4",
                10: "5",
                11: "6",
                12: "7",
                13: "8",
                14: "9",
                15: ";",
                16: "=",
                17: "B",
                18: "K",
                19: "N",
                20: "O",
                21: "Q",
                22: "R",
                23: "a",
                24: "b",
                25: "c",
                26: "d",
                27: "e",
                28: "f",
                29: "g",
                30: "h",
                31: "x",
            },
            "stoi": {
                " ": 0,
                "#": 1,
                "+": 2,
                "-": 3,
                ".": 4,
                "0": 5,
                "1": 6,
                "2": 7,
                "3": 8,
                "4": 9,
                "5": 10,
                "6": 11,
                "7": 12,
                "8": 13,
                "9": 14,
                ";": 15,
                "=": 16,
                "B": 17,
                "K": 18,
                "N": 19,
                "O": 20,
                "Q": 21,
                "R": 22,
                "a": 23,
                "b": 24,
                "c": 25,
                "d": 26,
                "e": 27,
                "f": 28,
                "g": 29,
                "h": 30,
                "x": 31,
            },
        }

        return base_meta

    # Initialize tokenizer metadata
    meta = initialize_tokenizer.__func__()

    @staticmethod
    def get_token_index(token):
        """
        Get the index of a token in the vocabulary.
        """
        return Globals.meta["stoi"].get(token, Globals.meta["stoi"]["<unk>"])

    @staticmethod
    def get_index_token(index):
        """
        Get the token at a given index in the vocabulary.
        """
        return Globals.meta["itos"].get(index, "<unk>")

if __name__ == "__main__":
    Globals.print()
