todo: move pokersnowie/eval_self_play.py to designated agent_folder/
# prl_baselines

## Installation
1. `git clone --recurse-submodules https://github.com/hellovertex/prl_environment.git`
2. `git clone https://github.com/hellovertex/prl_baselines.git`

### Inside virtual env run
Install Poker RL-environment as namespace package

2. `cd prl_environment`
3. `git submodule update --recursive --remote` (and `--init` if you did not clone using `--recurse-submodules`)
4. `pip install -e .`  `# use -e optionally for development`

Install Training data generation package
1. `cd ../prl_baselines`
2. `pip install -e .`  `# use -e optionally for development`

## Usage
This will create `.csv` files in the `data/folder`, each containing 500k training examples.
Note: Setting `--blind_sizes` parameter will determine the subfolder to which training data
is written, e.g. `data/folder/0.25-0.50/`. 

Possible values are
`'0.01-0.02'`, ... `'0.25-0.50', '0.50-1.00', '1.00-2.00', '10.00-20.00',` etc.

Hint 1: By passing `--unzipped_dir` flag, the `--zip_path` parameter is ignored and unzipping will 
be skipped. Instead training data will be generated from .txt files found in `unzipped_dir`

### First Approach - Training data from all players
1. `python data_acquisition/main.py ` pass `--from_gdrive_id` with a value of 
   1. `18kkgEM2CYF_Tl4Dn8oro6tUgqDfr9IAN` for the whole dataset (!) 5.3GB zipped and 60GB unzipped
   2. `18GE6Xw4K1XE2PNiXSyh762mJ5ZCRl2SO` for a small example zipfile
2. `python data_preprocessing/main.py `
3. `python training/main.py`

### Second Approach - Training data from subset of players (most active and best/worst earning )
1. Run `python eda.py` to generate player stats
2. Run `python data_acquistion/main.py --version_two` that reads set of players from
   `data/01_raw/{blind_size}/eda_result_filtered.txt` that was generated by `eda.py` 




## When running from Azure ML - VM
1. Before executing the `run_generate_train_data` script
 - **provide the .zip file as a dataset inside Azure ML**.

See `prl_docs\azure` for instructions on how to create a dataset from zipfile. 

2. Verify dataset availability inside Azure ML-VM Terminal:
- `az login`
- `az ml data list`
3. run `prl/baselines/supervised_learning/azure-notebooks/get_dataset_path.ipynb` to get
filepath to the zip file that is passed to `run_generate_train_data` via `--zip_path`


