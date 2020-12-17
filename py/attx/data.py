from typing import List, Tuple
import numpy as np
import copy

START_TOKEN = "STKN"
END_TOKEN = "ETKN"
NEXT_TOKEN = "NTKN"
NONE_TOKEN = "NONE"

SEQ_START = "A"
SEQ_END = "Z"
NUM_STD_CHARS = ord(SEQ_END) - ord(SEQ_START)
VOCAB = [chr(i) for i in range(ord(SEQ_START), ord(SEQ_END) + 1)] + [
    START_TOKEN,
    END_TOKEN,
    NEXT_TOKEN,
    NONE_TOKEN,
]
VOCAB_SIZE = len(VOCAB)
VOCAB_INDEX = {token: i for i, token in enumerate(VOCAB)}
VOCAB_INVERTED_INDEX = {i: token for i, token in enumerate(VOCAB)}


def create_sequence(length: int) -> List[str]:
    """
    Generates a sequence of letters from the alphabet.
    Cycles back the beginning after the last character.
    :param length: min value is three
    :return:
    """
    assert length >= 3, "Minimum sequence length is 3"
    random_starting_position = np.random.randint(0, NUM_STD_CHARS)

    values = [
        chr((token % NUM_STD_CHARS) + ord(SEQ_START))
        for token in range(random_starting_position, random_starting_position + length)
    ]

    return values


def create_example(sequence: List[str]) -> Tuple[List[str], str]:
    """
    Creates a problem example.
    This functions inserts a next token into the sequence, at random.
    The output of the problem, trivially, is to predict the character following
    the next sequence token.

    Example: given an input (A, B, C, D), the output can be:
        - (A, B, C, NTKN, D) -> D
        - (A, NTKN, B, C, D) -> B
    
    :param sequence: contains a sequence of characters, e.g. A, B, C, D.
    """
    assert len(sequence) >= 3, "Minimum sequence length is 3"
    next_token_position = np.random.randint(0 + 1, len(sequence) - 1)
    new_seq = copy.deepcopy(sequence)
    new_seq.insert(next_token_position, NEXT_TOKEN)
    return new_seq, new_seq[next_token_position + 1]
