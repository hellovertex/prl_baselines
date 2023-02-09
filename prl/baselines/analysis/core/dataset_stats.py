import re


class DatasetStats:

    def __init__(self):
        self.total_hands = 0
        self.total_showdowns = 0
        self.n_showdowns_no_mucks = 0
        self.n_showdowns_with_mucks = 0
        # actions are part of dataset stats
        # but these are not computed here
        # but after vectorizing the dataset observations
        # and discretizing the action space

    def _upd(self, hands_played):
        # todo increment total hands
        #  increment showdowns no mucks
        #  increment showdowns with mucks
        #  check for showdown in string and muckin string
        pass

    def update_from_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:  # pylint: disable=invalid-name,unspecified-encoding
            hand_database = f.read()
            hands_played = re.split(r'PokerStars Hand #', hand_database)[1:]
            self._upd(hands_played)

    def to_dict(self):
        return {'total_hands': self.total_hands,
                'total_showdowns': self.total_showdowns,
                'n_showdowns_no_mucks': self.n_showdowns_no_mucks,
                'n_showdowns_with_mucks': self.n_showdowns_with_mucks, }