import ast
import glob
import multiprocessing
import time
from functools import partial

import click
from prl.environment.Wrappers.augment import AugmentObservationWrapper

from csv_writer import CSVWriter
from hsmithy_parser import HSmithyParser
from prl.baselines.supervised_learning.config import LOGFILE
from prl.baselines.supervised_learning.data_acquisition.runner import Runner
from rl_state_encoder import RLStateEncoder


def get_output_name(selected_players, drop_folds, randomize_fold_cards):
    if selected_players:
        if drop_folds:
            return 'actions_selected_players__do_not_generate_fold_labels'  # Ds_nf
        elif randomize_fold_cards:
            return 'actions_selected_players__generate_folds__randomize_folded_cards'
        else:
            return 'actions_selected_players__generate_folds__keep_folded_cards'
    else:
        if drop_folds:
            return 'actions_all_winners__do_not_generate_fold_labels'
        elif randomize_fold_cards:
            return 'actions_all_winners__generate_folds__randomize_folded_cards'
        else:
            return 'actions_all_winners__generate_folds__keep_folded_cards'


def parse_encode_write(filename,
                       from_selected_players,
                       drop_folds,
                       randomize_fold_cards):
    blind_sizes = "0.25-0.50"
    parser = HSmithyParser()
    output_path = get_output_name(from_selected_players,
                             drop_folds,
                             randomize_fold_cards)
    # Steps Steinberger Poker Environment, augments observations and vectorizes them
    encoder = RLStateEncoder(env_wrapper_cls=AugmentObservationWrapper)

    # writes training data from encoder to disk
    writer = CSVWriter(out_filename_base=f'6MAX_{blind_sizes}')
    with open("/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/eda_result_filtered.txt",
              "r") as data:
        player_dict = ast.literal_eval(data.read())

    # Uses the results of parser and encoder to write training data to disk or cloud
    runner = Runner(parser=parser,
                    out_dir=blind_sizes + f'/{output_path}',  # todo change this name to output path
                    encoder=encoder,
                    writer=writer,
                    write_azure=False,
                    logfile=LOGFILE,
                    drop_folds = drop_folds,
                    randomize_fold_cards=randomize_fold_cards,
                    use_outdir_per_player=from_selected_players,
                    only_from_selected_players=from_selected_players,
                    selected_players=list(player_dict.keys()) if from_selected_players else None)
    # parse PokerEpisodes, encode, vectorize, write training data and labels to disk
    # unzipped_dir = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    return runner.parse_encode_write(filename)


def make_datasets(filenames, params, debug):
    # from_selected_players = False
    # drop_folds = False
    # randomize_folds_cards = False
    # the `params` define which name the output dataset folder has:
    # path_out_D_nf = 'actions_all_winners__do_not_generate_fold_labels'
    # path_out_D_f_nr = 'actions_all_winners__generate_folds__keep_folded_cards'
    # path_out_D_f_r = 'actions_all_winners__generate_folds__randomize_folded_cards'
    # path_out_Ds_nf = 'actions_selected_players__do_not_generate_folds'
    # path_out_Ds_f_nr = 'actions_selected_players__generate_folds__keep_folded_cards'
    # path_out_Ds_f_r = 'actions_selected_players__generate_folds__randomize_folded_cards'

    run_fn = partial(parse_encode_write, **params)
    if debug:
        run_fn(filename=filenames[16])
    else:
        print(f'Starting job. This may take a while.')
        start = time.time()
        p = multiprocessing.Pool()
        t0 = time.time()
        for x in p.imap_unordered(run_fn, filenames):
            print(x + f'. Took {time.time() - t0} seconds')
        print(f'Finished job after {time.time() - start} seconds.')
        p.close()


@click.command()
@click.option("--out_dir",
              default="0.25-0.50",
              type=str,
              help="Possible values are e.g. '0.25-0.50', '0.50-1.00', '1.00-2.00'")
@click.option("--from_gdrive_id",
              # for small example, use 18GE6Xw4K1XE2PNiXSyh762mJ5ZCRl2SO
              # for complete database (VERY LARGE 60GB unzipped), use 18kkgEM2CYF_Tl4Dn8oro6tUgqDfr9IAN
              default="18kkgEM2CYF_Tl4Dn8oro6tUgqDfr9IAN",
              type=str,
              help="Google drive id of a .zip file containing poker hands. "
                   "For small example, use 18GE6Xw4K1XE2PNiXSyh762mJ5ZCRl2SO"
                   "For complete database (VERY LARGE), use 18kkgEM2CYF_Tl4Dn8oro6tUgqDfr9IAN"
                   "The id can be obtained from the google drive download-link url."
                   "The runner will try to download the data from gdrive and proceed with unzipping."
                   "If unzipped_dir is passed as an argument, this parameter will be ignored.")
@click.option("--unzipped_dir",
              default="/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data",
              type=str,  # absolute path
              help="Absolute Path. Passing unzipped_dir we can bypass the unzipping step and assume "
                   "files have alredy been unzipped. ")
@click.option("--filter_selected_players",
              is_flag=True,
              default=False,
              help="See runner.run docstring for an explanation of what changed with version two.")
def main(blind_sizes, from_gdrive_id, unzipped_dir, version_two, use_player_names_as_outdir):
    # # extract zipfile with Poker-hands (.zip is stored locally or downloaded via from_gdrive)
    # if not unzipped_dir:
    #     unzipped_dir = self._extract_all_zip_data(from_gdrive_id)
    # filenames = glob.glob(unzipped_dir.__str__() + '/**/*.txt', recursive=True)
    # if from_gdrive_id:
    #     # try to download from_gdrive to out.zip
    #     zipfiles = [gdown.download(id=from_gdrive_id,
    #                                output=f"{path_to_data}/bulkhands_{self.out_dir}.zip",
    #                                quiet=False)]
    # else:
    #     #
    #     zipfiles = glob.glob(path_to_data.__str__() + '/*.zip', recursive=False)
    # out_dir = str(DATA_DIR) + "/01_raw/" + f'{self.out_dir}/unzipped'
    # # creates out_dir if it does not exist
    # # extracts zip file, only if extracted files with same name do not exist
    # [utils.extract(f_zip, out_dir=out_dir) for f_zip in zipfiles]
    # return out_dir

    """Extracts .zip files found in prl_baselines/data/01_raw -- unless `unzipped_dir` is provided.
     Reads the extracted .txt files and 1) parses, 2) encodes, 3) vectorizes poker hands and 4) writes them to disk.
     The .zip file can also be downloaded from gdrive by providing a gdrive-url."""
    # selected_players: pass list of players to use only showdowns with at least on of em in
    # use_fold: set to true to generate fold labels
    # -- if selected players is set, fold labels will be the other players actions + their cards are randomized
    # the following will only be necessary for 2NL for which we have no time left
    # todo: assume data is present per player folder
    #  that means we can ignore 2NL completely for now
    # -- if not selected players, the fold labels will be the losing players action + their cards are randomized
    # data present per player (0.25NL) or total files (2NL)
    # main()
    # unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/2.5NL/unzipped"
    unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data"
    filenames = glob.glob(unzipped_dir.__str__() + '/**/*.txt', recursive=True)
    # so many parameters...
    # use_multiprocessing
    # selected_players, drop_folds, randomize_fold_cards  ---> can determine out_name
    # blind sizes
    #
    debug = True
    for from_selected_players in [True, False]:
        for drop_folds in [True, False]:
            if drop_folds:
                for randomize_folds_cards in [True, False]:
                    # parse encode write here
                    params = {'from_selected_players': from_selected_players,
                              'drop_folds': drop_folds,
                              'randomize_fold_cards': randomize_folds_cards}
                    make_datasets(filenames, params, debug)
            else:
                params = {'from_selected_players': from_selected_players,
                          'drop_folds': drop_folds,
                          'randomize_fold_cards': False}
                make_datasets(filenames, params, debug)


if __name__ == '__main__':
    main()
