import logging
import argparse
import sys
import os
from sklearn.datasets import dump_svmlight_file
import numpy as np
import pickle
import csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--thresh-hold', type=int, default=0, help='an integer for the map')
    parser.add_argument('--dataset-type', type=str, default='raw', help='an integer for the map')
    args = parser.parse_args()

    logger = logging.getLogger('FE')
    logger.setLevel(logging.INFO)
    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setFormatter(logging.Formatter(fmt='{asctime}:{levelname}:{name}:{message}', style='{'))
    logger.addHandler(log_handler)

    curr_dir = os.path.dirname(__file__)
    in_train_path = os.path.join(curr_dir, 'output', 'train_' + args.dataset_type + '_' + str(args.thresh_hold) + '.csv')
    in_valid_path = os.path.join(curr_dir, 'output',
                                 'valid_' + args.dataset_type + '_' + str(args.thresh_hold) + '.csv')
    out_train_path = os.path.join(curr_dir, 'output', 'trainsvm_' + args.dataset_type + '_' + str(args.thresh_hold) + '.libsvm')
    out_valid_path = os.path.join(curr_dir, 'output',
                                 'validsvm_' + args.dataset_type + '_' + str(args.thresh_hold) + '.libsvm')

    reversed_map_path = os.path.join(curr_dir, 'info', 'reversed_map_' + str(args.thresh_hold) + '.pickle')
    feature_lens = dict()
    with open(in_train_path, 'r') as f:
        column_names = f.readline().strip().split(',')
        assert column_names[0] == 'id'
        assert column_names[-1] == 'weekday'
        assert len(column_names) == 25
    with open(reversed_map_path, 'rb') as f:
        reversed_map_dict = pickle.load(f)
        for k, v in reversed_map_dict.items():
            feature_lens[k] = len(v)
        del reversed_map_dict

    start_pos = 0
    feature_start = list()
    for column_name in column_names[2:]:
        feature_start.append(start_pos)
        if column_name == 'hour':
            start_pos += 24
        elif column_name == 'weekday':
            start_pos += 7
        else:
            start_pos += feature_lens[column_name]
    logger.info(str(feature_lens))
    logger.info(str(feature_start))

    # ds = np.genfromtxt(in_train_path, delimiter=',', names=column_names, skip_header=True, dtype=np.int32, excludelist=['id'])
    with open(in_train_path, 'r') as fin, open(out_train_path, 'w') as fout:
        fin.readline()
        for line_no, line in enumerate(fin):
            line = line.strip().split(',')[1:]
            new_line = [line[0]]
            new_line.extend(map(lambda x: str(x[0] + int(x[1])) + ':1', zip(feature_start, line[1:])))
            fout.write(' \t'.join(new_line) + '\n')
            if (line_no % 400000) == 0:
                logger.info('progress %d', line_no)
        logger.info('progress %d, done ', line_no)

    with open(in_valid_path, 'r') as fin, open(out_valid_path, 'w') as fout:
        fin.readline()
        for line_no, line in enumerate(fin):
            line = line.strip().split(',')[1:]
            new_line = [line[0]]
            new_line.extend(map(lambda x: str(x[0] + int(x[1])) + ':1', zip(feature_start, line[1:])))
            fout.write(' \t'.join(new_line) + '\n')
            if (line_no % 400000) == 0:
                logger.info('progress %d', line_no)
        logger.info('progress %d, done ', line_no)
