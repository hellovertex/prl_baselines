"""Use this on virutal machines to unzip BulkHands.zip.
Useful before running select_players.py"""
import glob
import io
import os
import zipfile

import click


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


@click.command()
@click.option("--path_to_zipfile",
              default="",
              type=str,
              help="Absolute path ending with .zip")
@click.option("--abs_out_path",
              default="",
              type=str,  # absolute path
              help="Absolute path where .zip file should be unzipped to.")
def main(path_to_zipfile, abs_out_path):
    zipfiles = glob.glob(path_to_zipfile, recursive=False)
    [extract(zipfile, out_dir=abs_out_path) for zipfile in zipfiles]


if __name__ == '__main__':
    main()
