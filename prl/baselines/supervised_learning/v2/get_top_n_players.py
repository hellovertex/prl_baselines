# In: n, reversed=False  # if true, looks for the worst `n` players
# Out: Dict[str, PlayerStats]
import glob

from prl.baselines.supervised_learning.v2.fast_hsmithy_parser import ParseHsmithyTextToPokerEpisode


def main(target_dir, n, reversed=False):
    # automatically searches target_dir recursively.
    # if you want to browse individual files, move them first
    filenames = glob.glob(target_dir+' **/*.txt', recursive=True)
    parser = ParseHsmithyTextToPokerEpisode()
    for filename in filenames:
        episodes = parser.parse_file(filename)
        pass


if __name__ == "__main__":
    unzipped_dir = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    n = 100
    main(unzipped_dir, n)