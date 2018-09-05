import os, sys
import logging
import collections
import pickle
import csv

if __name__ == '__main__':
    logger = logging.getLogger('FE')
    logger.setLevel(logging.INFO)
    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setFormatter(logging.Formatter(fmt='{asctime}:{levelname}:{name}:{message}', style='{'))
    logger.addHandler(log_handler)

    progress_step = 400000

    train_dump_filename = 'train_count.pickle'
    valid_dump_filename = 'valid_count.pickle'
    test_dump_filename = 'test_count.pickle'

    curr_dir = os.path.dirname(__file__)
    output_dir = os.path.join(curr_dir, 'info')
    train_filepath = os.path.join(curr_dir, 'output', 'train.csv')
    test_filepath = os.path.join(curr_dir, 'output', 'test.csv')
    train_dump_path = os.path.join(output_dir, train_dump_filename)
    valid_dump_path = os.path.join(output_dir, valid_dump_filename)
    test_dump_path = os.path.join(output_dir, test_dump_filename)

    train_count_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    valid_count_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    test_count_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))

    with open(train_filepath, 'r') as f:
        reader = csv.DictReader(f)
        for line_no, feature_dict in enumerate(reader):
            feature_dict.pop('id')
            if feature_dict['hour'][4:6] != '30':
                for feature_name, feature in feature_dict.items():
                    train_count_dict[feature_name][feature] += 1
            else:
                for feature_name, feature in feature_dict.items():
                    valid_count_dict[feature_name][feature] += 1
            if (line_no % progress_step) == 0:
                logger.info('train valid progress %d', line_no)
        logger.info('train valid progress %d, done!!!', line_no)

    with open(test_filepath, 'r') as f:
        reader = csv.DictReader(f)
        for line_no, feature_dict in enumerate(reader):
            feature_dict.pop('id')
            for feature_name, feature in feature_dict.items():
                test_count_dict[feature_name][feature] += 1
            if (line_no % progress_step) == 0:
                logger.info('test progress %d', line_no)
        logger.info('test progress %d, done!!!', line_no)

    train_count_dict = dict(map(lambda x: (x[0], dict(x[1])), train_count_dict.items()))
    valid_count_dict = dict(map(lambda x: (x[0], dict(x[1])), valid_count_dict.items()))
    test_count_dict = dict(map(lambda x: (x[0], dict(x[1])), test_count_dict.items()))

    with open(train_dump_path, 'wb') as f:
        pickle.dump(train_count_dict, f, pickle.HIGHEST_PROTOCOL)

    with open(valid_dump_path, 'wb') as f:
        pickle.dump(valid_count_dict, f, pickle.HIGHEST_PROTOCOL)

    with open(test_dump_path, 'wb') as f:
        pickle.dump(test_count_dict, f, pickle.HIGHEST_PROTOCOL)
