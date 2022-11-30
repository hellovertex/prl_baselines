def test_flush_better_than_three_pair():
    # https://www.poker.org/poker-hands-ranking-chart/
    pass


def test_three_pair_better():
    # https://www.poker.org/poker-hands-ranking-chart/
    pass


def test_mc_hand_completion():
    hero_cards_1d = [33, 9]
    board_cards_1d = [32, 11, 17]
    n_opponents = 2
    deck = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51]
    len_deck = 48