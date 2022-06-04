import glob

import click


@click.command
@click.option('--input_dir',
              type=str,
              help='location of .csv files containing vectorized information')
@click.option('--output_dir',
              type=str,
              help='location of where to write .parquet files [optional]')
@click.option('--use_downsampling',
              is_flag=True,
              show_default=True,
              default=True,
              help='Whether to balance labels in loaded .csv files. '
                   'Enabling this is recommended for training')
@click.option('--save_to_parquet',
              is_flag=True,
              show_default=True,
              default=True,
              help='Whether the loaded dataframes should be written to .parquet files.'
                   'Parquet files consume less memory.')
def main(input_dir, save_to_parquet, use_downsampling, output_dir):
    csv_files = glob.glob(input_dir.__str__() + '/**/*.txt', recursive=True)


if __name__ == '__main__':
    main()
