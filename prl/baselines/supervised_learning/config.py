import os
from pathlib import Path
# do not move this file, relative paths would break
DATA_DIR = Path(os.path.abspath(__file__)).parent.parent.parent.parent.joinpath('data')
LOGFILE = DATA_DIR.joinpath("log.txt")
