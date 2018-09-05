import os
import sys
import logging
import pickle
from datetime import datetime
import argparse
import collections
import multiprocessing

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--thresh-hold', type=int, default=0, help='an integer for the map')
    args = parser.parse_args()

    logger = logging.getLogger('FE')
    logger.setLevel(logging.INFO)
    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setFormatter(logging.Formatter(fmt='{asctime}:{levelname}:{name}:{message}', style='{'))
    logger.addHandler(log_handler)

    curr_dir = os.path.dirname(__file__)
    feature_map_path = os.path.join(curr_dir, 'info', 'feature_map_' + str(args.thresh_hold) + '.pickle')
    reversed_map_path = os.path.join(curr_dir, 'info', 'reversed_map_' + str(args.thresh_hold) + '.pickle')

    train_path = os.path.join(curr_dir, 'output', 'train_raw_' + str(args.thresh_hold) + '.csv')
    valid_path = os.path.join(curr_dir, 'output', 'valid_raw_' + str(args.thresh_hold) + '.csv')

    train_filepath = os.path.join(curr_dir, 'output', 'train.csv')

    logger.info('load feature map')
    with open(feature_map_path, 'rb') as f:
        feature_map_dict = pickle.load(f)

    with open(train_filepath, 'r') as ftrain:
        column_names = ftrain.readline().strip().split(',')

    def get_new_line(line):
        line = line.strip().split(',')
        new_line = [line[0], line[1]]
        parsed_time = datetime.strptime('20' + line[2], '%Y%m%d%H')
        new_line.append(str(parsed_time.hour))
        for feature_name, feature in zip(column_names[3:], line[3:]):
            new_line.append(str(feature_map_dict[feature_name][feature]))
        new_line.append(str(parsed_time.weekday()))
        return ','.join(new_line) + '\n'

    train_split_list = []
    valid_split_list = []
    logger.info('load data set')
    with open(train_filepath, 'r') as ftrain:
        header = ftrain.readline()
        for line_no, line in enumerate(ftrain):
            timestamp = line.strip().split(',')[2]
            if timestamp[4:6] != '30':
                train_split_list.append(line)
            else:
                valid_split_list.append(line)

            if (line_no % 400000) == 0:
                logger.info('progress %d', line_no)
        logger.info('load progress %d, done!!!', line_no)

    logger.info('write train file')
    header = header.strip() + ',weekday\n'
    with open(train_path, 'w') as ftrain_out:
        ftrain_out.write(header)
        with multiprocessing.Pool(processes=7) as p:
            for line in p.map(get_new_line, train_split_list, chunksize=10000):
                ftrain_out.write(line)

    logger.info('write valid file')
    with open(valid_path, 'w') as fvalid_out:
        fvalid_out.write(header)
        with multiprocessing.Pool(processes=7) as p:
            for line in p.map(get_new_line, valid_split_list, chunksize=10000):
                fvalid_out.write(line)
