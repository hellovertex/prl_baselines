import numpy as np
from typing import Union
import numba as nb

dict_stein_to_sk = {0: 49, 1: 50, 2: 48, 3: 51, 4: 45, 5: 46, 6: 44, 7: 47, 8: 41, 9: 42, 10: 40, 11: 43, 12: 37, 13: 38,
               14: 36, 15: 39, 16: 33, 17: 34, 18: 32, 19: 35, 20: 29, 21: 30, 22: 28, 23: 31, 24: 25, 25: 26, 26: 24,
               27: 27, 28: 21, 29: 22, 30: 20, 31: 23, 32: 17, 33: 18, 34: 16, 35: 19, 36: 13, 37: 14, 38: 12, 39: 15,
               40: 9, 41: 10, 42: 8, 43: 11, 44: 5, 45: 6, 46: 4, 47: 7, 48: 1, 49: 2, 50: 0, 51: 3}

dict_str_to_sk = {'As': 0, 'Ah': 1, 'Ad': 2, 'Ac': 3, 'Ks': 4, 'Kh': 5, 'Kd': 6, 'Kc': 7, 'Qs': 8, 'Qh': 9,
                             'Qd': 10, 'Qc': 11, 'Js': 12, 'Jh': 13, 'Jd': 14, 'Jc': 15, 'Ts': 16, 'Th': 17, 'Td': 18,
                             'Tc': 19, '9s': 20, '9h': 21, '9d': 22, '9c': 23, '8s': 24, '8h': 25, '8d': 26, '8c': 27,
                             '7s': 28, '7h': 29, '7d': 30, '7c': 31, '6s': 32, '6h': 33, '6d': 34, '6c': 35, '5s': 36,
                             '5h': 37, '5d': 38, '5c': 39, '4s': 40, '4h': 41, '4d': 42, '4c': 43, '3s': 44, '3h': 45,
                             '3d': 46, '3c': 47, '2s': 48, '2h': 49, '2d': 50, '2c': 51}


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
