import numpy as np
from typing import Union
import numba as nb

stein_to_sk = {0: 49, 1: 50, 2: 48, 3: 51, 4: 45, 5: 46, 6: 44, 7: 47, 8: 41, 9: 42, 10: 40, 11: 43, 12: 37, 13: 38,
               14: 36, 15: 39, 16: 33, 17: 34, 18: 32, 19: 35, 20: 29, 21: 30, 22: 28, 23: 31, 24: 25, 25: 26, 26: 24,
               27: 27, 28: 21, 29: 22, 30: 20, 31: 23, 32: 17, 33: 18, 34: 16, 35: 19, 36: 13, 37: 14, 38: 12, 39: 15,
               40: 9, 41: 10, 42: 8, 43: 11, 44: 5, 45: 6, 46: 4, 47: 7, 48: 1, 49: 2, 50: 0, 51: 3}


def int_steinberger_card_to_int_skpokereval_card(c_stein: int):
    return stein_to_sk[c_stein]


def arr1d_steinberger_cards_to_arr1d_skpokereval_cards(cards: Union[list, np.ndarray]):
    return [int_steinberger_card_to_int_skpokereval_card(c) for c in cards]


def test_arr1d_conversion():
    arr = np.array([0, 1, 2, 30, 40, 50, 51])
    assert arr1d_steinberger_cards_to_arr1d_skpokereval_cards(arr) == [49, 50, 48, 20, 9, 0, 3]
    arr = [0, 1, 2, 30, 40, 50, 51]
    assert arr1d_steinberger_cards_to_arr1d_skpokereval_cards(arr) == [49, 50, 48, 20, 9, 0, 3]


if __name__ == '__main__':
    test_arr1d_conversion()
