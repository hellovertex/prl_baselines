"""


fonts_ds = tf.data.experimental.make_csv_dataset(
    file_pattern = "fonts/*.csv",
    batch_size=10, num_epochs=1,
    num_parallel_reads=20,
    shuffle_buffer_size=10000)



# https://www.tensorflow.org/tutorials/load_data/csv#tfiodecode_csv
train_dataset = tf.data.Dataset.list_files(PATH+"/*.csv")
print(train_dataset)
train_dataset = train_dataset.map(lambda x: tf.py_function(load,[x],[tf.float32]) , num_parallel_calls=tf.data.experimental.AUTOTUNE)

"""
from functools import partial

import tensorflow as tf
import csv
import glob
import pathlib
import os, sys, codecs
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)

NCOLS = 564
DEFAULT_RECORDS = [0. for _ in range(NCOLS)]


def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    print("Execution time:", time.perf_counter() - start_time)


def get_num_cols(csv_filepath):
    with open(csv_filepath, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        return len(next(reader))


def _load(csv_filepath, ncols):
    record_defaults = tf.constant(DEFAULT_RECORDS, dtype=tf.float32)
    # record_defaults = tf.constant(['' for _ in range(ncols)], dtype=tf.float32)
    f = pd.read_csv(csv_filepath)
    data = tf.convert_to_tensor(f, dtype=tf.float32)
    return data


def remove_bom_inplace_if_applicable(path):
    """Removes BOM mark, if it exists, from a file and rewrites it in-place"""
    buffer_size = 4096
    bom_length = len(codecs.BOM_UTF8)

    with open(path, "r+b") as fp:
        chunk = fp.read(buffer_size)
        if chunk.startswith(codecs.BOM_UTF8):
            i = 0
            chunk = chunk[bom_length:]
            while chunk:
                fp.seek(i)
                fp.write(chunk)
                i += len(chunk)
                fp.seek(bom_length, os.SEEK_CUR)
                chunk = fp.read(buffer_size)
            fp.seek(-bom_length, os.SEEK_CUR)
            fp.truncate()


def make_csv_ds(path):
    record_defaults = tf.constant(DEFAULT_RECORDS, dtype=tf.float32)
    return tf.data.experimental.CsvDataset(
        path,
        record_defaults=record_defaults,
        header=True)


def create_dataset(input_dir) -> tf.data.Dataset:
    # load function
    unzipped_filenames = glob.glob(input_dir.__str__() + '/**/*.csv', recursive=True)
    # [remove_bom_inplace_if_applicable(f) for f in unzipped_filenames]
    dataset = tf.data.Dataset.from_tensor_slices(unzipped_filenames)

    def parse_fn(filename):
        return tf.data.Dataset.range(10)

    def pack_row(*row):
        label = row[0]
        features = tf.stack(row[1:], 1)
        return features, label


    record_defaults = tf.constant(DEFAULT_RECORDS, dtype=tf.float32)
    ds = dataset.interleave(lambda x:
                            # tf.data.experimental.CsvDataset(x, record_defaults=record_defaults, header=True), #.map(parse_fn, num_parallel_calls=1),
                            tf.data.TextLineDataset(x), #.map(parse_fn, num_parallel_calls=1),
                            cycle_length=4, block_length=16)

    # ds_files = tf.data.Dataset.list_files(input_dir.__str__() + '/*.csv', )
    # print(ds_files)
    # record_defaults = tf.constant([0. for _ in range(NCOLS)], dtype=tf.float32)
    # ncols = get_num_cols(unzipped_filenames[1])
    # fn = partial(make_csv_ds, record_defaults)
    # ds = ds_files.interleave(lambda x: tf.py_function(fn, x), cycle_length=3)
    # ncols = get_num_cols(unzipped_filenames[0])
    # load_fn = partial(_load, ncols=ncols)
    # record_defaults = tf.constant([0. for _ in range(ncols)], dtype=tf.float32)
    # # dataset loaded lazily
    # train_dataset = tf.data.Dataset.from_tensor_slices(unzipped_filenames)
    # train_dataset = train_dataset.map(lambda x: tf.py_function(load_fn, x, [tf.float32]),
    #                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ds = tf.data.experimental.CsvDataset(
    #     filenames=unzipped_filenames,
    #     record_defaults=record_defaults,
    #     header=True,
    #     batch_size=10, num_epochs=1,
    #     num_parallel_reads=20,
    #     shuffle_buffer_size=10000)

    return ds



def train_eval_fn(input_dir):
    # *** TRAIN ***
    # 1.from training dir, read csv files lazily with shuffling
    # https://www.tensorflow.org/guide/data
    dataset = create_dataset(input_dir)
    print(tf.data.Dataset.element_spec)
    for elem in dataset.batch(10).take(10):
        print(elem)
        print(elem.numpy())

    # 2a. using best practices from tf dataset summary
    # https://www.tensorflow.org/guide/data_performance?hl=en#best_practice_summary
    # measure train/test error, loss
    # https: // www.tensorflow.org / tutorials / customization / custom_training_walkthrough  # evaluate_the_model_on_the_test_dataset
    # checkpoints / save model
    # https://www.tensorflow.org/guide/checkpoint?hl=en
    # save summary
    # https://www.tensorflow.org/tensorboard/get_started#tensorboarddev_host_and_share_your_ml_experiment_results

    # *** EVAL ***
    # measure test error
    # https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough#evaluate_the_model_on_the_test_dataset
    # load model
    # https://www.tensorflow.org/guide/saved_model?hl=en

    # *** ANALYSIS ***
    # analyse td.data performance (tf profiler)
    # 2.b https://www.tensorflow.org/guide/data_performance_analysis?hl=en
    # analyse tf performance
    # https://www.tensorflow.org/guide/profiler?hl=en
    # analyse gpu performance
    # https://www.tensorflow.org/guide/gpu_performance_analysis?hl=en

    pass
