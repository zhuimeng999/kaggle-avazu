# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Prepare MovieLens dataset for wide-deep."""

import tensorflow as tf
import os
import pickle


def get_feature_dim(thresh_hold):
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    feature_dim_path = os.path.join(curr_dir, 'feature_dim_' + str(thresh_hold) + '.pickle')
    with open(feature_dim_path, 'rb') as f:
        feature_dim_dict = pickle.load(f)
    return feature_dim_dict


def get_input_fn(train_path, batch_size, repeat, shuffle):
    with open(train_path, 'r') as f:
        column_names = f.readline().strip().split(',')
    _CSV_COLUMN_DEFAULTS = [[0]]*(len(column_names) - 1)

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS, 
                                select_cols=list(range(len(column_names)))[1:])
        features = dict(zip(column_names[1:], columns))
        # features.pop('id')
    
        labels = features.pop('click')
        return features, labels

    def csv_input_fn():
        # Extract lines from input files using the Dataset API.
        dataset = tf.data.TextLineDataset(train_path)
        dataset = dataset.skip(1)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle)

        # We call repeat after shuffling, rather than before, to prevent separate
        # epochs from blending together.
        if repeat > 1:
            dataset = dataset.repeat(repeat)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(parse_csv, num_parallel_calls=None)
        return dataset.prefetch(2)

    return csv_input_fn


def write_to_file(thresh_hold):
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    reversed_map_path = os.path.join(curr_dir, '../../preprocess/info/reversed_map_' + str(thresh_hold) + '.pickle')
    with open(reversed_map_path, 'rb') as f:
        reversed_map_dict = pickle.load(f)

    feature_dim_dict = { k: len(v) for k, v in reversed_map_dict.items()}
    feature_dim_dict['hour'] = 24
    feature_dim_dict['weekday'] = 7
    assert len(feature_dim_dict) == 23

    feature_dim_path = os.path.join(curr_dir, 'feature_dim_' + str(thresh_hold) + '.pickle')
    with open(feature_dim_path, 'wb') as f:
        pickle.dump(feature_dim_dict, f, pickle.HIGHEST_PROTOCOL)


def main(_):
    write_to_file(0)
    write_to_file(100)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
