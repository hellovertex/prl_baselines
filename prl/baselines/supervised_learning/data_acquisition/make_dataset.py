import os
from typing import List

DATA_DIR = os.environ['PRL_BASELINES_DATA_DIR']


def create_dataset_from_players(folder_with_text_files: str,
                                selected_players: List[str],
                                out_folder: str):
    # filenames =
    # parsed_episodes = parser.parse_file(abs_filepath)  # line 140 runner.py
    #
    return 0


if __name__ == '__main__':
    folder_with_text_files = DATA_DIR + '/01_raw/0.25-0.50/unzipped'
    selected_players = ["ishuha"]
    out_folder = DATA_DIR + '01_raw/selected_players'
    dataset = create_dataset_from_players(folder_with_text_files,
                                          selected_players,
                                          out_folder)
