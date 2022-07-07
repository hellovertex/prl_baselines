import glob
import io
import os
import zipfile

import gdown
import numpy as np
import pandas as pd

from core.encoder import Encoder
from core.generator import TrainingDataGenerator
from core.parser import Parser

import sqlite3


# from azureml.core import Experiment
# from azureml.core import Workspace


class SteinbergerGenerator(TrainingDataGenerator):
    """This handles creation and population of training data folders, containing encoded
    PokerEpisode instances. These encodings can be used for supervised learning. """

    def __init__(self, data_dir: str,
                 parser: Parser,
                 encoder: Encoder,
                 out_filename: str,
                 write_azure: bool,
                 logfile="log.txt"):
        self._out_filename = out_filename
        self._data_dir = data_dir
        self._parser = parser
        self._encoder = encoder
        self._write_azure = write_azure
        self._experiment = None
        self._logfile = logfile
        self._n_files_written_this_run = 0
        self._num_lines_written = 0
        self._blind_sizes = None
        self._hand_counter = 0
        self._n_invalid_files = 0

        with open(logfile, "a+") as f:
            self._n_files_already_encoded = len(f.readlines())
            print(f'reinitializing with {self._n_files_already_encoded} files already encoded')

        # if write_azure:
        #     self._experiment = Experiment(workspace=self.get_workspace(),
        #                                   name="supervised-baseline")

    def __enter__(self):
        if self._write_azure:
            self._run = self._experiment.start_logging()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # print(exc_type, exc_value, exc_traceback)
        if self._write_azure:
            self._run.complete()

    # @staticmethod
    # def get_workspace():
    #     config = {
    #         "subscription_id": "0d263aec-21a1-4c68-90f8-687d99ccb93b",
    #         "resource_group": "thesis",
    #         "workspace_name": "generate-train-data"
    #     }
    #     # connect to get-train-data workspace
    #     return Workspace.get(name=config["workspace_name"],
    #                          subscription_id=config["subscription_id"],
    #                          resource_group=config["resource_group"])

    @staticmethod
    def file_has_been_encoded_already(logfile, filename: str):
        skip_file = False
        # skip already encoded files
        with open(logfile, "a+") as f:
            files_written = f.readlines().__reversed__()
            for fw in files_written:
                if filename in fw:
                    print(f"Skipping file {filename} because it has already been encoded and written to disk...")
                    skip_file = True
                    break
        return skip_file

    @property
    def out_filename(self):
        return self._out_filename

    def _write_to_azure(self, abs_filepath):
        self._run.upload_file(name="output.csv", path_or_stream=abs_filepath)

    def _log_progress(self, abs_filepath):
        with open(self._logfile, "a") as f:
            f.write(abs_filepath + "\n")
        self._n_files_written_this_run += 1

    def extract(self, filename, out_dir):
        z = zipfile.ZipFile(filename)
        for f in z.namelist():
            try:
                os.mkdir(out_dir)
            except FileExistsError:
                pass
            # read inner zip file into bytes buffer
            content = io.BytesIO(z.read(f))
            zip_file = zipfile.ZipFile(content)
            for i in zip_file.namelist():
                zip_file.extract(i, out_dir)

    def _extract_all_zip_data(self, zip_path, blind_sizes, from_gdrive_id):
        path_to_data = self._data_dir + "01_raw/" + blind_sizes
        if from_gdrive_id:
            # try to download from_gdrive to out.zip
            zipfiles = [gdown.download(id=from_gdrive_id,
                                       output=f"{path_to_data}/bulkhands_{blind_sizes}.zip",
                                       quiet=False)]
        else:
            #
            zipfiles = glob.glob(zip_path.__str__(), recursive=False)
        out_dir = self._data_dir + "01_raw/" + f'{self._blind_sizes}/unzipped'
        # creates out_dir if it does not exist
        # extracts zip file, only if extracted files with same name do not exist
        [self.extract(zipfile, out_dir=out_dir) for zipfile in zipfiles]
        return out_dir

    def _write_metadata(self, file_dir):
        file_path_metadata = os.path.join(file_dir, f"{self._out_filename}.meta")
        with open(file_path_metadata, "a") as file:
            file.write(self._parser.metadata.__repr__() + "\n")
        return file_path_metadata

    def _write_train_data(self, data, labels):
        raise NotImplementedError

    def _generate_training_data(self, from_parsed_hands):
        training_data, labels = None, None
        for i, hand in enumerate(from_parsed_hands):
            observations, actions = self._encoder.encode_episode(hand)
            if not observations:
                continue
            if training_data is None:
                training_data = observations
                labels = actions
            else:
                try:
                    training_data = np.concatenate((training_data, observations), axis=0)
                    labels = np.concatenate((labels, actions), axis=0)
                except Exception as e:
                    print(e)
            self._num_lines_written += len(observations)
            print("Simulating environment", end='') if i == 0 else print('.', end='')
            self._hand_counter += 1
        return training_data, labels

    def generate_from_file(self, abs_filepath):
        """Docstring"""
        self._hand_counter = 0
        try:
            parsed_hands = self._parser.parse_file(abs_filepath)
        except UnicodeDecodeError:
            print('---------------------------------------')
            print(f'Skipping {self._n_invalid_files}th invalid file {abs_filepath} because it has invalid continuation byte...')
            print('---------------------------------------')
            self._n_skipped += 1  # todo push fix
            return
        training_data, labels = self._generate_training_data(parsed_hands)

        # some rare cases, where the file did not contain showdown plays
        if training_data is not None:
            print(f"\nExtracted {len(training_data)} training samples from {self._hand_counter + 1} poker hands"
                  f"in file {self._n_files_written_this_run + self._n_files_already_encoded} {abs_filepath}...")

            self._log_progress(abs_filepath)
            # write train data
            file_dir, file_path = self._write_train_data(training_data, labels)

            # write to cloud
            if self._write_azure:
                self._write_to_azure(file_path)

            # write meta data
            file_path_metadata = self._write_metadata(file_dir=file_dir)
            # df = pd.read_csv(file_path)
            # print(f"Data created: and written to {file_path}, "
            #       f"metadata information is found at {file_path_metadata}")
            # print(df.head())

    def run_data_generation(self, zip_path, blind_sizes, from_gdrive_id, unzipped_dir=None):
        self._blind_sizes = blind_sizes
        # extract zipfile (.zip is stored locally or downloaded via from_gdrive)
        if not unzipped_dir:
            unzipped_dir = self._extract_all_zip_data(zip_path, blind_sizes, from_gdrive_id)
        filenames = glob.glob(unzipped_dir.__str__() + '/**/*.txt', recursive=True)
        # parse, encode, vectorize and write the training data from .txt to disk
        for i, filename in enumerate(filenames):
            if not self.file_has_been_encoded_already(logfile=self._logfile,
                                                      filename=filename):
                self.generate_from_file(os.path.abspath(filename).__str__())


class CsvTrainingDataGenerator(SteinbergerGenerator):
    """This handles creation and population of training data folders, containing encoded
    PokerEpisode instances. These encodings can be used for supervised learning. """

    def __init__(self, data_dir: str,
                 parser: Parser,
                 encoder: Encoder,
                 out_filename: str,
                 write_azure: bool,
                 logfile="log.txt"):
        super().__init__(data_dir,
                         parser,
                         encoder,
                         out_filename,
                         write_azure,
                         logfile)

    def _write_train_data(self, data, labels):
        file_dir = os.path.join(self._data_dir + "02_vectorized", self._blind_sizes)
        # create new file every 500k lines
        file_name = self._out_filename + '_' + str(int(self._num_lines_written / 500000)) + '.csv'
        file_path = os.path.join(file_dir, file_name)
        columns = None
        if not os.path.exists(file_path):
            os.makedirs(os.path.realpath(file_dir), exist_ok=True)
            columns = self._encoder.feature_names
        pd.DataFrame(data=data,
                     index=labels,
                     columns=columns).to_csv(
            file_path, index_label='label', mode='a')
        return file_dir, file_path


class ParquetTrainingDataGenerator(SteinbergerGenerator):
    """This handles creation and population of training data folders, containing encoded
    PokerEpisode instances. These encodings can be used for supervised learning. """

    def __init__(self, data_dir: str,
                 parser: Parser,
                 encoder: Encoder,
                 out_filename: str,
                 write_azure: bool,
                 logfile="log.txt"):
        super().__init__(data_dir,
                         parser,
                         encoder,
                         out_filename,
                         write_azure,
                         logfile)

    def _write_train_data(self, data, labels):
        """todo: Implement writing to parquet file."""
        # todo can we append without recreating the file?
