import click

from prl.baselines.pokersnowie.eighteighteight import EightEightEightConverter
from prl.baselines.pokersnowie.generate_database import HandHistorySmithyToPokerSnowie
from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser


@click.command()
@click.option("--path_in",
              default="",
              type=str,
              help="Absolute path to hhsmithy .txt databases")
@click.option("--path_out",
              default="",
              type=str,
              help="Absolute path where PokerSnowie database result should be written to")
@click.option("--n_hands", default=500000,
              type=int,
              help="How many PokerSnowie hands should be written to a single .txt file")
@click.option("--selected_players_file",
              default="",
              type=str,
              help="Absolute path where dict with selected players is stored. "
                   "See select_players.py to generate it")
def main(path_in, path_out, n_hands, selected_players_file):
    """Translates databases from https://www.hhsmithy.com/ to https://www.pokersnowie.com/ databases.
    These are .txt files in human-readable format """
    # Parses hhsmithy databases
    parser = HSmithyParser()

    # Converts parsed hhsmithy databases to PokerSnowie databases using 888-Format
    converter = EightEightEightConverter()

    # writes PokerSnowie databses to .txt files
    db_gen = HandHistorySmithyToPokerSnowie(parser=parser, converter=converter)
    db_gen.generate_database(path_in, path_out, n_hands, selected_players_file)


if __name__ == '__main__':
    main()
