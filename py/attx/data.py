from typing import List, Tuple
import numpy as np
import copy

START_TOKEN = "BTKN"
END_TOKEN = "ETKN"
NEXT_TOKEN = "NTKN"

SEQ_START = "A"
SEQ_END = "Z"
SEQ_MAX_LEN = ord(SEQ_END) - ord(SEQ_START)


def create_sequence(length: int) -> List[str]:
    """
    Generates a sequence of letters from the alphabet.
    :param length: min value is three
    :return:
    """
    assert length >= 3, "Minimum sequence length is 3"
    random_starting_position = np.random.randint(0, SEQ_MAX_LEN)

    values = [
        chr((token % SEQ_MAX_LEN) + ord(SEQ_START))
        for token in range(random_starting_position, random_starting_position + length)
    ]

    return values


def create_example(sequence: List[str]) -> Tuple[List[str], str]:
    assert len(sequence) >= 3, "Minimum sequence length is 3"
    next_token_position = np.random.randint(0 + 1, len(sequence) - 1)
    new_seq = copy.deepcopy(sequence)
    new_seq.insert(next_token_position, NEXT_TOKEN)
    return new_seq, new_seq[next_token_position + 1]
