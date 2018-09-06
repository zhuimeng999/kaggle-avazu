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

def get_input_fn(train_path, batch_size, repeat, shuffle):
    with open(train_path, 'r') as f:
        column_names = f.readline().strip().split(',')
    _CSV_COLUMN_DEFAULTS = [[0]]*(len(column_names) - 1)
    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS, 
                                select_cols=list(range(len(column_names)))[1:])
        features = dict(zip(column_names[1:], columns))
        #features.pop('id')
    
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
