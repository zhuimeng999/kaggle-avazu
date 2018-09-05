import hashlib
import logging
import pickle
import os
import sys
import argparse

NO_MAP_FEATURE = 'id', 'click', 'hour'
ONE_HOT_ENCODED_FEATURES = ['C1', 'banner_pos', 'site_category', 'app_category', 'device_type', 'device_conn_type',
                            'C15', 'C16', 'C18', 'C20']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--thresh-hold', type=int, default=0, help='an integer for the map')
    args = parser.parse_args()

    logger = logging.getLogger('FE')
    logger.setLevel(logging.INFO)
    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setFormatter(logging.Formatter(fmt='{asctime}:{levelname}:{name}:{message}', style='{'))
    logger.addHandler(log_handler)

    logger.info('load feature count dict ...')

    train_dump_filename = 'train_count.pickle'
    valid_dump_filename = 'valid_count.pickle'
    test_dump_filename = 'test_count.pickle'

    curr_dir = os.path.dirname(__file__)
    train_dump_path = os.path.join(curr_dir, 'info', train_dump_filename)
    valid_dump_path = os.path.join(curr_dir, 'info', valid_dump_filename)
    test_dump_path = os.path.join(curr_dir, 'info', test_dump_filename)
    with open(train_dump_path, 'rb') as f:
        train_count_dict = pickle.load(f)

    with open(valid_dump_path, 'rb') as f:
        valid_count_dict = pickle.load(f)

    with open(test_dump_path, 'rb') as f:
        test_count_dict = pickle.load(f)

    logger.info('load feature count dict done...')

    logger.info('build feature map dict ...')
    reversed_map_dict = {feature_name: list() for feature_name in train_count_dict if feature_name not in NO_MAP_FEATURE}
    feature_map_dict = {feature_name: dict() for feature_name in train_count_dict if feature_name not in NO_MAP_FEATURE}

    logger.info('mapped feature total %d', len(reversed_map_dict))
    logger.info(str(list(reversed_map_dict.keys())))

    logger.info('one hot feature ...')
    for feature_name in ONE_HOT_ENCODED_FEATURES:
        v = train_count_dict[feature_name]
        for index, (feature, count) in enumerate(sorted(v.items(), reverse=True, key=lambda x: x[1])):
            feature_map_dict[feature_name][feature] = index
            reversed_map_dict[feature_name].append([feature])
        logger.info('build feature map: feature name %s, mapped size %d', feature_name, index)

    logger.info('Truncated feature ...')
    for feature_name, v in train_count_dict.items():
        if (feature_name in ONE_HOT_ENCODED_FEATURES) or (feature_name in NO_MAP_FEATURE):
            continue
        curr_total = 0
        curr_index = 0
        reversed_map_dict[feature_name].append(list())
        for feature, count in sorted(v.items(), reverse=True, key=lambda x: x[1]):
            curr_total += count
            feature_map_dict[feature_name][feature] = curr_index
            reversed_map_dict[feature_name][curr_index].append(feature)
            if curr_total > args.thresh_hold:
                curr_total = 0
                curr_index += 1
                reversed_map_dict[feature_name].append(list())
        if not reversed_map_dict[feature_name][curr_index]:
            assert curr_total == 0
            del reversed_map_dict[feature_name][curr_index]

        for feature in valid_count_dict[feature_name]:
            if feature not in feature_map_dict[feature_name]:
                feature_encoded = (feature_name + '_' + feature).encode()
                index = int(hashlib.md5(feature_encoded).hexdigest(), 16) % len(reversed_map_dict[feature_name])
                feature_map_dict[feature_name][feature] = index
                reversed_map_dict[feature_name][index].append(feature)
        logger.info('build feature map: feature name %s, mapped size %d', feature_name, curr_index)
    logger.info('build feature map dict done...')

    logger.info('save pickle ...')

    feature_map_path = os.path.join(curr_dir, 'info', 'feature_map_' + str(args.thresh_hold) + '.pickle')
    reversed_map_path = os.path.join(curr_dir, 'info', 'reversed_map_' + str(args.thresh_hold) + '.pickle')

    with open(feature_map_path, 'wb') as f:
        pickle.dump(feature_map_dict, f, pickle.HIGHEST_PROTOCOL)

    with open(reversed_map_path, 'wb') as f:
        pickle.dump(reversed_map_dict, f, pickle.HIGHEST_PROTOCOL)
    logger.info('save pickle done...')



