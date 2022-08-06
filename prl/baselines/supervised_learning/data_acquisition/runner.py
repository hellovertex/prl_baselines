import glob
import io
import os
import zipfile

import gdown
import numpy as np

from prl.baselines.supervised_learning.config import DATA_DIR
from prl.baselines.supervised_learning.data_acquisition.core.encoder import Encoder
from prl.baselines.supervised_learning.data_acquisition.core.parser import Parser
from prl.baselines.supervised_learning.data_acquisition.core.writer import Writer


class Runner:
    def __init__(self,
                 parser: Parser,
                 encoder: Encoder,
                 writer: Writer,
                 write_azure: bool,
                 logfile="log.txt"):
        self.parser = parser
        self.encoder = encoder
        self.writer = writer
        self.logfile = logfile
        self.write_azure = write_azure

        self._n_files_written_this_run = 0
        self._hand_counter = 0
        self._n_invalid_files = 0
        self._n_skipped = 0
        self._experiment = None
        self.blind_sizes = None

        with open(logfile, "a+") as f:
            self._n_files_already_encoded = len(f.readlines())
            print(f'reinitializing with {self._n_files_already_encoded} files already encoded')

    def __enter__(self):
        if self.write_azure:
            self._run = self._experiment.start_logging()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # print(exc_type, exc_value, exc_traceback)
        if self.write_azure:
            self._run.complete()

    @staticmethod
    def file_has_been_encoded_already(logfile, filename: str):
        skip_file = False
        # skip already encoded files
        with open(logfile, "a+") as f:
            files_written = f.readlines().__reversed__()
            for fw in files_written:
                if filename in fw:
                    print(
                        f"Skipping file {filename} because it has already been encoded and written to disk...")
                    skip_file = True
                    break
        return skip_file

    def _write_metadata(self, file_dir):
        file_path_metadata = os.path.join(file_dir, f"{self.writer.out_filename_base}.meta")
        with open(file_path_metadata, "a") as file:
            file.write(self.parser.metadata.__repr__() + "\n")
        return file_path_metadata

    def _encode(self, from_parsed_hands):
        training_data, labels = None, None
        n_samples = 0
        for i, hand in enumerate(from_parsed_hands):
            observations, actions = self.encoder.encode_episode(hand)
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
            n_samples += len(observations)
            print("Simulating environment", end='') if i == 0 else print('.', end='')
            self._hand_counter += 1
        return training_data, labels, n_samples

    @staticmethod
    def extract(filename, out_dir):
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

    def _extract_all_zip_data(self, from_gdrive_id=None) -> str:
        """
        :param from_gdrive_id: google drive id of .zip file. If None, .zip file will be looked up in data folder.
        :return: directory to which poker data has been unzipped
        """
        path_to_data = str(DATA_DIR) + "/01_raw/" + self.blind_sizes
        if from_gdrive_id:
            # try to download from_gdrive to out.zip
            zipfiles = [gdown.download(id=from_gdrive_id,
                                       output=f"{path_to_data}/bulkhands_{self.blind_sizes}.zip",
                                       quiet=False)]
        else:
            #
            zipfiles = glob.glob(path_to_data.__str__() + '/*.zip', recursive=False)
        out_dir = str(DATA_DIR) + "/01_raw/" + f'{self.blind_sizes}/unzipped'
        # creates out_dir if it does not exist
        # extracts zip file, only if extracted files with same name do not exist
        [self.extract(zipfile, out_dir=out_dir) for zipfile in zipfiles]
        return out_dir

    def parse(self, abs_filepath):
        parsed_hands = None
        try:
            parsed_hands = self.parser.parse_file(abs_filepath)
        except UnicodeDecodeError:
            print('---------------------------------------')
            print(
                f'Skipping {self._n_invalid_files}th invalid file {abs_filepath} because it has invalid continuation byte...')
            print('---------------------------------------')
            self._n_skipped += 1  # todo push fix
        return parsed_hands

    def parse_encode_write(self, abs_filepath):
        """Docstring"""
        # parse
        parsed_hands = self.parse(abs_filepath)
        # encode
        training_data, labels, n_samples = self._encode(parsed_hands)
        # write
        if training_data is not None:
            print(f"\nExtracted {len(training_data)} training samples from {self._hand_counter + 1} poker hands"
                  f"in file {self._n_files_written_this_run + self._n_files_already_encoded} {abs_filepath}...")

            self.writer.log_progress(self.logfile, abs_filepath)
            # write train data
            file_dir, file_path = self.writer.write_train_data(training_data,
                                                               labels,
                                                               self.encoder.feature_names,
                                                               n_samples, self.blind_sizes)

            self._write_metadata(file_dir=file_dir)

    def run(self, blind_sizes, unzipped_dir=None, from_gdrive_id=None):
        """
        :param blind_sizes: determines data folder paths that are looked up and created, e.g. "data/01_raw/0.25-0.50"
        :param unzipped_dir: if a zip file has been unpacked previously, pass folder containing its unzipped content
        :param from_gdrive_id: if the zip file is not stored locally, it can be downloaded from gdrive-url
        """
        self.blind_sizes = blind_sizes
        self._hand_counter = 0
        # extract zipfile (.zip is stored locally or downloaded via from_gdrive)
        if not unzipped_dir:
            unzipped_dir = self._extract_all_zip_data(from_gdrive_id)
        filenames = glob.glob(unzipped_dir.__str__() + '/**/*.txt', recursive=True)
        # parse, encode, vectorize and write the training data from .txt to disk
        for i, filename in enumerate(filenames):
            if not self.file_has_been_encoded_already(logfile=self.logfile, filename=filename):
                self.parse_encode_write(os.path.abspath(filename).__str__())






