# coding: utf-8
# pylint: disable = invalid-name, C0111
import json
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle
import os, sys
import logging
import argparse
from io import TextIOBase


def prepare_work():
    search_work_dir = 'leave_l1_l2'
    if os.path.exists(search_work_dir):
        print('path is already exist !!!')
        exit()
    os.makedirs(search_work_dir)
    os.chdir(search_work_dir)
    
    logger = logging.getLogger('lightgbm')
    logger.setLevel(logging.INFO)
    my_format = logging.Formatter(fmt='{asctime}:{levelname}:{name}:{message}', style='{')
    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setFormatter(my_format)
    logger.addHandler(log_handler)
    log_handler = logging.StreamHandler(open('search.log', 'w+'))
    log_handler.setFormatter(my_format)
    logger.addHandler(log_handler)
    
    class tmp_stdout(TextIOBase):
        def write(self, value):
            logger.info(value.replace('\r', ''))
    class tmp_stderr(TextIOBase):
        def write(self, value):
            logger.info(value)
    sys.stdout = tmp_stdout()
    sys.stderr = tmp_stderr()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--thresh-hold', type=int, default=0, help='an integer for the map')
    args = parser.parse_args()
    
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(curr_dir)
    prepare_work()
    logger = logging.getLogger('lightgbm')
    
    TRAIN_DATA_PATH = os.path.join(curr_dir, '../../preprocess/output/train_raw_0.csv')
    VALID_DATA_PATH = os.path.join(curr_dir, '../../preprocess/output/train_raw_0.csv')
    
    search_work_dir = 'leave_l1_l2'
    if os.path.exists(search_work_dir):
        print('path is already exist !!!')
        exit()
    os.makedirs(search_work_dir)
    
    with open(TRAIN_DATA_PATH, 'r') as f:
        column_names = f.readline().strip().split(',')
    print(column_names)
    # load or create your dataset
    print('Load data...')
    # create dataset for lightgbm
    # if you want to re-use data, remember to set free_raw_data=False
    column_names.remove('click')
    dataset_params = {
        'free_raw_data': False,
        'silent': False,
        'feature_name': column_names,
        'categorical_feature': None,
        'params': {
            'header': True,
            'label_column': 'name:click',
            'ignore_column': 'name:id',
            }
        }
    lgb_train = lgb.Dataset(TRAIN_DATA_PATH, **dataset_params)
    lgb_valid = lgb.Dataset(VALID_DATA_PATH,reference=lgb_train, **dataset_params)

    num_leaves_grid = [63, 127, 255, 511]
    lambda_l1_grid = [0, 0.001, 0.01, 0.1]
    lambda_l2_grid = [0, 0.001, 0.01, 0.1]
    
    
    grid_search_list = []
    for num_leaves in num_leaves_grid:
        for lambda_l1 in lambda_l1_grid:
            for lambda_l2 in lambda_l2_grid:
                grid_search_list.append(
                    {'num_leaves':num_leaves, 'lambda_l1': lambda_l1, 'lambda_l2':lambda_l2})
    for s_dict in grid_search_list:
                logger.info('current params: '+ str(s_dict))
                # specify your configurations as a dict
                model_params = {
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'metric_freq': 1,
                    'is_training_metric': True,
                    'num_leaves': s_dict['num_leaves'],
                    'learning_rate': 0.1,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'num_leaves': 255,
                    'num_bin': 255,
                    'bin_construct_sample_cnt': 3,
                    'min_data_in_leaf': 50,
                    'min_sum_hessian_in_leaf': 5.0,
                    'use_two_round_loading = false': False,
                    'lambda_l1': s_dict['lambda_l1'],
                    'lambda_l2': s_dict['lambda_l2'],
                    'verbose': 0
                }
            
                print('Start training...')
                # feature_name and categorical_feature
                gbm = lgb.train(model_params,
                                lgb_train,
                                num_boost_round=100,
                                valid_sets=lgb_valid,  # eval training data
                                categorical_feature=column_names[1:]
                                )
            
                # save model to file
                print('save model to file...')
                
                model_name = 'model_'
                model_name = model_name + str(s_dict['num_leaves']) + '_'
                model_name = model_name + str(s_dict['lambda_l1']) + '_' 
                model_name = model_name + str(s_dict['lambda_l2']) + '.txt'
                gbm.save_model(model_name)
