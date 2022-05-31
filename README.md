# prl_baselines

## How to: Fresh Start
1. `git clone https://github.com/hellovertex/prl_environment.git`
2. `git clone https://github.com/hellovertex/prl_baselines.git`

### Inside virtual env run
Install Poker RL-environment as namespace package
3. `pip install requests`
4. `cd prl_environment`
5. `git submodule update --init`
6. `pip install -e .`  `# use -e optionally for development`

Install Training data generation package
7. `cd ../prl_baselines`
8. `pip install -e .`  `# use -e optionally for development`

Run training data generation from .zipfile to .csv files
9. `python run_generate_train_data --zip_path "<PATH_TO_ZIPFILE_WITH_BULKHANDS>"`

This will create `.csv` files in the `data/folder`, each containing 500k training examples.
Note: Setting `--blind_sizes` parameter will determine the subfolder to which training data
is written, e.g. `data/folder/0.25-0.50/`. 

Possible values are
`'0.01-0.02'`, ... `'0.25-0.50', '0.50-1.00', '1.00-2.00', '10.00-20.00',` etc.

## When running from Azure ML - VM
1. Before executing the `run_generate_train_data` script
 - **provide the .zip file as a dataset inside Azure ML**.

See `prl_docs\azure` for instructions on how to create a dataset from zipfile. 

2. Verify dataset availability inside Azure ML-VM Terminal:
- `az login`
- `az ml data list`
3. run `prl/baselines/supervised_learning/azure-notebooks/get_dataset_path.ipynb` to get
filepath to the zip file that is passed to `run_generate_train_data` via `--zip_path`


