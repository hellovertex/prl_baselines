"""Use this on virutal machines to get the BulkHands.zip from google drive.
Useful before running eda.py"""
import click
import gdown


@click.option("--from_gdrive_id",
              default="",
              type=str,
              help="Google drive id of a bulkhands.zip file containing poker hands. "
                   "The id can be obtained from the google drive download-link url."
                   "The runner will try to download the data from gdrive and proceed with unzipping."
                   "If unzipped_dir is passed as an argument, this parameter will be ignored.")
@click.option("--abs_out_path",
              default="",
              type=str,  # absolute path
              help="Absolute path where downloaded .zip file should go. Must end with .zip")
def main(from_gdrive_id, abs_out_path):
    gdown.download(id=from_gdrive_id,
                   output=abs_out_path,  # must end with .zip
                   quiet=False)


if __name__ == '__main__':
    main()