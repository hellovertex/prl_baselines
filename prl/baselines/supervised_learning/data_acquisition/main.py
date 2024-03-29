import ast
import glob
import multiprocessing
import time
from functools import partial
from pathlib import Path

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
                    drop_folds=drop_folds,
                    randomize_fold_cards=randomize_fold_cards,
                    use_outdir_per_player=from_selected_players,
                    only_from_selected_players=from_selected_players,
                    selected_players=list(player_dict.keys()) if from_selected_players else None)
    # parse PokerEpisodes, encode, vectorize, write training data and labels to disk
    # unzipped_dir = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    return runner.parse_encode_write(filename)


def make_datasets_selected_players_from_Dprime(filenames, params, debug):
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


def make_datasets_all_players_from_D(filename,
                                     debug,
                                     from_selected_players,
                                     drop_folds,
                                     randomize_fold_cards
                                     ):
    blind_sizes = "0.25-0.50"
    # out_filename_base = f'6MAX_{blind_sizes}_{filenames[-1]}'
    # out_filename_base = f'6MAX_{blind_sizes}_{filenames[-1]}'
    parser = HSmithyParser()
    output_path = get_output_name(from_selected_players,
                                  drop_folds,
                                  randomize_fold_cards)
    # Steps Steinberger Poker Environment, augments observations and vectorizes them
    encoder = RLStateEncoder(env_wrapper_cls=AugmentObservationWrapper)

    # parse PokerEpisodes, encode, vectorize, write training data and labels to disk
    # unzipped_dir = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    # if debug:
    #     pass
    # #     for filename in filenames[:1]:
    # #         runner.parse_encode_write(filename)
    # else:
    #     for filename in filenames[:-1]:
    out_filename_base = f'6MAX_{blind_sizes}_{Path(filename).stem}'
    # writes training data from encoder to disk
    writer = CSVWriter(out_filename_base=out_filename_base)
    with open(
            "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/eda_result_filtered.txt",
            "r") as data:
        player_dict = ast.literal_eval(data.read())

    # Uses the results of parser and encoder to write training data to disk or cloud
    runner = Runner(parser=parser,
                    out_dir=blind_sizes + f'/{output_path}',  # todo change this name to output path
                    encoder=encoder,
                    writer=writer,
                    write_azure=False,
                    logfile=LOGFILE,
                    drop_folds=drop_folds,
                    randomize_fold_cards=randomize_fold_cards,
                    use_outdir_per_player=from_selected_players,
                    only_from_selected_players=from_selected_players,
                    selected_players=list(player_dict.keys()) if from_selected_players else None)
    runner.parse_encode_write(filename)
    return "Success."


def run_make_datasets(fn, filenames):
    start = time.time()
    p = multiprocessing.Pool()
    # run f0
    for x in p.imap_unordered(fn, filenames):
        print(x + f'. Took {time.time() - start} seconds')
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
# @click.option("--filter_selected_players",
#               is_flag=True,
#               default=False,
#               help="See runner.run docstring for an explanation of what changed with version two.")
def main(out_dir, from_gdrive_id, unzipped_dir):
    # unzpped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    # unzinames = glob.glob(unzipped_dir.__str__() + '/**/*.txt', recursive=True)
    # fileipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data"
    debug = True

    # unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data"
    # filenames = glob.glob(unzipped_dir.__str__() + '/**/*.txt', recursive=True)
    # params_0 = {'from_selected_players': True,
    #             'drop_folds': True,
    #             'fold_random_cards': False}
    # params_1 = {'from_selected_players': True,
    #             'drop_folds': False,
    #             'fold_random_cards': False}
    # params_2 = {'from_selected_players': True,
    #             'drop_folds': False,
    #             'fold_random_cards': True}
    # make_datasets_selected_players_from_Dprime(filenames, params_0, debug)
    # make_datasets_selected_players_from_Dprime(filenames, params_1, debug)
    # make_datasets_selected_players_from_Dprime(filenames, params_2, debug)

    # todo consider making extra function for this repeated snippet
    debug = False
    unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data_336"
    filenames = glob.glob(unzipped_dir.__str__() + '/**/*.txt', recursive=True)
    params_0 = {'from_selected_players': False,
                'drop_folds': False,
                'fold_random_cards': False}
    params_1 = {'from_selected_players': False,
                'drop_folds': False,
                'fold_random_cards': True}
    params_2 = {'from_selected_players': False,
                'drop_folds': True,
                'fold_random_cards': False  # this will be ignored when `drop_folds=true`
                }
    # x = 10000  # for unzipped
    x = 12  # for 336 player data
    chunks = []
    current_chunk = []
    i = 0
    for file in filenames:
        current_chunk.append(file)
        if (i + 1) % x == 0:
            chunks.append(current_chunk)
            current_chunk = []
        i += 1
    # trick to avoid multiprocessing writes to same file
    for i, chunk in enumerate(chunks):
        chunk.append(f'CHUNK_INDEX_{i}')
    run_fn_0 = partial(make_datasets_all_players_from_D, **params_0, debug=debug)
    run_fn_1 = partial(make_datasets_all_players_from_D, **params_1, debug=debug)
    run_fn_2 = partial(make_datasets_all_players_from_D, **params_2, debug=debug)
    # run_make_datasets(run_fn_0, chunks=chunks)
    run_make_datasets(run_fn_1, filenames=filenames)
    # run_make_datasets(run_fn_2, chunks=chunks)
    print("ALL DONE")


if __name__ == '__main__':
    main()
