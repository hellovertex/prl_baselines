# from __future__ import annotations

import argparse
import os
import sys

# import mlflow
# import mlflow.pytorch
import mlflow
import torch
import torch.optim as optim
from azureml.core import Workspace, Dataset, Model, Run
# from tensorboardX import SummaryWriter

from dataset import get_dataloaders
from model import Net, train, test

# DATA_DIR = '../../../../data/'

aml_run = Run.get_context()


class Args(object):
    pass


# Training settings
args = Args()
setattr(args, 'batch_size', 512)
setattr(args, 'test_batch_size', 1000)
setattr(args, 'epochs', 100)
setattr(args, 'lr', 1e-6)
setattr(args, 'momentum', 0.5)
setattr(args, 'use_cuda', True)
setattr(args, 'seed', 1)
setattr(args, 'log_interval', 10)
setattr(args, 'log_artifacts_interval', 10)
setattr(args, 'save_model', True)
setattr(args, 'output_dir', "artifacts")
use_cuda = True  # not args.use_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda")

kwargs = {'num_workers': 1, 'pin_memory': True}


# def extract(filename, out_dir):
#     z = zipfile.ZipFile(filename)
#     for f in z.namelist():
#         try:
#             os.mkdir(out_dir)
#         except FileExistsError:
#             pass
#         # read inner zip file into bytes buffer
#         content = io.BytesIO(z.read(f))
#         zip_file = zipfile.ZipFile(content)
#         for i in zip_file.namelist():
#             zip_file.extract(i, out_dir)


def run(train_loader, test_loader, model_name):
    ws = Workspace.from_config()
    # mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    # with mlflow.start_run() as run:
    model_dir = "outputs"
    model_path = "outputs/model.pkl"
    model_exists = False
    # model_name = "baseline_model"
    try:
        # # does not work apparently in the #*(&#@$ azure cloud
        # model = torch.load('./outputs/model.pt')
        # get the registered model
        # todo this probably does not work
        Model(ws, model_name).download(model_dir)
        model_exists = True
        model = torch.load(model_path)

    except Exception as e:
        # print(e)
        input_dim = output_dim = hidden_dim = None
        for data, label in train_loader:
            input_dim = data.shape[1]
            output_dim = label.shape[0]
            hidden_dim = [512, 512]
            break
        model = Net(input_dim, output_dim, hidden_dim)
    if args.use_cuda:
        model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    # Create a SummaryWriter to write TensorBoard events locally
    try:
        os.mkdir(model_dir)
    except FileExistsError:
        # model path already exists
        pass

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(epoch, args, model, test_loader, train_loader, device)
        # # does not work apparently in the #*(&#@$ azure cloud
        
        # register the model
        # model_uri = "runs:/{}/model".format(aml_run.info.run_id)
        # model = mlflow.register_model(model_uri, "baseline_model")
        # torch.save(model, model_path)
        # model = aml_run.register_model(model_name='baseline_model', model_path=model_path)
        # https://stackoverflow.com/questions/70928761/azureml-model-register
        # torch.save(model, model_path)
        if epoch % 10 == 0:
            Model.register(
                workspace=ws,
                model_name=model_name,
                model_path=model_path,
                model_framework=Model.Framework.PYTORCH,
                model_framework_version=torch.__version__
            )

            
    # return run
""""
todo fix this
Message: UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.supervised_baseline_training_1647868680_a100e6fa/pytorch-model/MLmodel already exists.
UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.supervised_baseline_training_1647868680_a100e6fa/pytorch-model/conda.yaml already exists.
UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.supervised_baseline_training_1647868680_a100e6fa/pytorch-model/requirements.txt already exists.
UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.supervised_baseline_training_1647868680_a100e6fa/pytorch-model/data/pickle_module_info.txt already exists.
UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.supervised_baseline_training_1647868680_a100e6fa/pytorch-model/data/model.pth already exists.
"""

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Provide model pipeline with necessary arguments: "
                    "- path to training data "
                    "-whatever else comes to my mind later")
    argparser.add_argument('-t', '--train_dir',
                           help='abs or rel path to .txt files with raw training samples.')
    argparser.add_argument('-m', '--model_name',
                           help='Name of Registered azure ml model. Will be loaded if exists.')
    # todo pass path to setup.py and make setup return mount context here
    # todo mount context can depend on _Offline or Run context
    cmd_args, _ = argparser.parse_known_args()
    # train_dir = cmd_args.train_dir
    # ws = Workspace.from_config()
    # ds = ws.get_default_datastore()
    # ds_paths = [(ds, '0.25-0.50/')]
    # dataset = Dataset.File.from_files(path=ds_paths)
    # print(f'dataset.to_path = {dataset.to_path()}')
    train_dir = sys.argv[2]

    print("===== DATA =====")
    print("DATA PATH: " + train_dir)
    print("LIST FILES IN DATA DIR...")
    print(os.listdir(train_dir))
    print("================")

    dataloaders = get_dataloaders(train_dir)
    train_loader = dataloaders['train']
    test_loader = dataloaders['test']

    run(train_loader, test_loader, model_name=cmd_args.model_name)
