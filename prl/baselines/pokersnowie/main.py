import click
from prl.environment.Wrappers.prl_wrappers import AugmentObservationWrapper

from prl.baselines.pokersnowie.generate_database import HandHistorySmithyToPokerSnowie
from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser
from prl.baselines.supervised_learning.config import LOGFILE


@click.command()
@click.option("--path_in",
              default="",
              type=str,
              help="Absolute path to hhsmithy .txt databases")
@click.option("--path_out",default="",
              type=str,
              help="Absolute path where PokerSnowie database result should be written to")
def main(path_in, path_out):
    """Translates databases from https://www.hhsmithy.com/ to https://www.pokersnowie.com/ databases
    Databases are .txt files in human readable format """
    # Parses hhsmithy databases
    parser = HSmithyParser()

    # Translates to PokerSnowie databases
    db_gen = HandHistorySmithyToPokerSnowie(parser=parser)

    # writes PokerSnowie databses to .txt files
    db_gen.generate_database(path_in, path_out)


if __name__ == '__main__':
    main()
