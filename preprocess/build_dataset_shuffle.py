import os
import sys
import logging
import argparse


def shuffle_dataset(raw_path, shuffle_path, step):
    logger = logging.getLogger('FE')
    with open(raw_path, 'r') as f:
        lines = f.readlines()
    line_total = len(lines)
    logger.info('shuffle %s line total %d', raw_path, line_total)
    with open(shuffle_path, 'w') as f:
        for i in range(step):
            j = i
            logger.info('shuffle step %d!!!', i)
            while j < line_total:
                f.write(lines[j])
                j += step
        logger.info('shuffle done!!!')


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

    train_raw_path = os.path.join(curr_dir, 'output', 'train_raw_' + str(args.thresh_hold) + '.csv')
    valid_raw_path = os.path.join(curr_dir, 'output', 'valid_raw_' + str(args.thresh_hold) + '.csv')

    train_path = os.path.join(curr_dir, 'output', 'train_shuffle_' + str(args.thresh_hold) + '.csv')
    valid_path = os.path.join(curr_dir, 'output', 'valid_shuffle_' + str(args.thresh_hold) + '.csv')

    shuffle_dataset(train_raw_path, train_path, 24*9)
    # no need
    # shuffle_dataset(valid_raw_path, valid_path)



