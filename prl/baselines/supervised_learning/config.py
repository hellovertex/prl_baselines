import os
from pathlib import Path
DATA_DIR = Path(os.path.abspath(__file__)).parent.parent.parent.parent.joinpath('data')
LOGFILE = DATA_DIR.joinpath("log.txt")
