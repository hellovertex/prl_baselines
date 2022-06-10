# prl_baselines

## How to: Fresh Start
1. `git clone --recurse-submodules https://github.com/hellovertex/prl_environment.git`
2. `git clone https://github.com/hellovertex/prl_baselines.git`

### Inside virtual env run
Install Poker RL-environment as namespace package

2. `cd prl_environment`
3. `git submodule update --recursive --remote` (and `--init` if you did not clone using `--recurse-submodules`)
4. `pip install -e .`  where using `-e` is optional for convenient development

Install Training data generation package
1. `cd ../prl_baselines`
2. `pip install -e .`  `# use -e optionally for development`

Run training data generation from .zipfile to .csv files
1. `python run_generate_train_data --zip_path "<PATH_TO_ZIPFILE_WITH_BULKHANDS>"`

This will create `.csv` files in the `data/folder`, each containing 500k training examples.
Note: Setting `--blind_sizes` parameter will determine the subfolder to which training data
is written, e.g. `data/folder/0.25-0.50/`. 

Possible values are
`'0.01-0.02'`, ... `'0.25-0.50', '0.50-1.00', '1.00-2.00', '10.00-20.00',` etc.

Tip 1: As the BulkHands.zip downlaoded from hsmithy can be very large (200.000 files unzipped),
it is recommended to generate multiple smaller zip files and run `run_generate_train_data` 
for each of these, possibly on different machines simultaneously.

Tip 2: By passing `--unzipped_dir` flag, the `--zip_path` parameter is ignored and unzipping will 
be skipped. Instead training data will be generated from .txt files found in `unzipped_dir`



## When running from Azure ML - VM
1. Before executing the `run_generate_train_data` script
 - **provide the .zip file as a dataset inside Azure ML**.

See `prl_docs\azure` for instructions on how to create a dataset from zipfile. 

2. Verify dataset availability inside Azure ML-VM Terminal:
- `az login`
- `az ml data list`
3. run `prl/baselines/supervised_learning/azure-notebooks/get_dataset_path.ipynb` to get
filepath to the zip file that is passed to `run_generate_train_data` via `--zip_path`


