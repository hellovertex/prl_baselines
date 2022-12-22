todo: move pokersnowie/eval_self_play.py to designated agent_folder/
# prl_baselines

----
Purpose of this repository:
1. [x] Supervised Learning of Poker Baseline Agent from Game Logs for our RL training
2. [x] Evaluation of resulting models using [PokerSnowie](https://www.pokersnowie.com/) 
3. [ ] todo: Compute PokerStats (Vpip, 3bet, cbet, etc) and compute evaluation metrics
4. [ ] todo: Create Monte-Carlo based Reference baseline that has no Neural Network component
##  Installation
1. `git clone --recurse-submodules https://github.com/hellovertex/prl_environment.git`
2. `git clone --recurse-submodules https://github.com/hellovertex/prl_baselines.git`

### Inside virtual env run
Install Poker RL-environment as namespace package

2. `cd prl_environment`
3. `git submodule update --recursive --remote` (and `--init` if you did not clone using `--recurse-submodules`)
4. `pip install -e .`  `# use -e optionally for development`

Install Baseline agent package (also generates training data from poker game logs)
1. `cd ../prl_baselines`
2. `git submodule update --recursive --remote` (and `--init` if you did not clone using `--recurse-submodules`)
2. `pip install -e .`  `# use -e optionally for development`

## Usage
### Supervised Learning of Poker Baseline Agent from Game Logs for our RL training
- [ ] todo: update following description to be more complete

This will create `.csv` files in the `data/folder`, each containing 500k training examples.
Note: Setting `--blind_sizes` parameter will determine the subfolder to which training data
is written, e.g. `data/folder/0.25-0.50/`. 

Possible values are
`'0.01-0.02'`, ... `'0.25-0.50', '0.50-1.00', '1.00-2.00', '10.00-20.00',` etc.

Hint 1: By passing `--unzipped_dir` flag, the `--zip_path` parameter is ignored and unzipping will 
be skipped. Instead training data will be generated from .txt files found in `unzipped_dir`

#### First Approach - Training data from all players
1. `python data_acquisition/main.py ` pass `--from_gdrive_id` with a value of 
   1. `18kkgEM2CYF_Tl4Dn8oro6tUgqDfr9IAN` for the whole dataset (!) 5.3GB zipped and 60GB unzipped
   2. `18GE6Xw4K1XE2PNiXSyh762mJ5ZCRl2SO` for a small example zipfile
2. `python data_preprocessing/main.py `
3. `python training/main.py`

#### Second Approach - Training data from subset of players (most active and best/worst earning )
1. Run `python eda.py` to generate player stats
2. Run `python data_acquistion/main.py --version_two` that reads set of players from
   `data/01_raw/{blind_size}/eda_result_filtered.txt` that was generated by `eda.py`

### Evaluation of resulting models using [PokerSnowie](https://www.pokersnowie.com/)
To generate pokersnowie databases from an Experiment-instance (see `prl/baselines/evaluation/core.experiment`),
you want to 
- run `python prl/baselines/evaluation/example_eval_with_pokersnowie.py`
.
- Before doing that, please set output path inside `prl/baselines/config.gin`.


