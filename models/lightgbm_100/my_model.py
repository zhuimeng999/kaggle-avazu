# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import os
import logging

curr_dir = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger('lightgbm grid search')


class MyModel(object):
    def __init__(self):
        self.TRAIN_DATA_PATH = os.path.join(curr_dir, '../../preprocess/output/train_raw_100.csv')
        self.VALID_DATA_PATH = os.path.join(curr_dir, '../../preprocess/output/valid_raw_100.csv')
        self.dataset_train = None
        self.dataset_valid = None
        self.model = None
        self.column_names = None

    def load_dataset(self):
        with open(self.TRAIN_DATA_PATH, 'r') as f:
            column_names = f.readline().strip().split(',')
        # load or create your dataset
        logger.info('Load data...')
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
                'max_bin': 255,
            }
        }
        self.column_names = column_names
        logger.info('clolumn_names:' + str(self.column_names))
        self.dataset_train = lgb.Dataset(self.TRAIN_DATA_PATH, **dataset_params)
        self.dataset_valid = lgb.Dataset(self.VALID_DATA_PATH, reference=self.dataset_train, **dataset_params)

    def train(self, **params):
        model_params = {
            'categorical_feature': self.column_names[1:],
            'num_boost_round': 100,
            'early_stopping_rounds': 5,
            'params': {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'binary_logloss',
                'metric_freq': 1,
                'is_provide_training_metric': True,
                'learning_rate': 0.1,
                'max_bin':255,
                'num_leaves': 255,
                'tree_learner': 'serial',
                'feature_fraction': 0.9,
                'bagging_freq': 5,
                'bagging_fraction': 0.8,
                'bin_construct_sample_cnt': 3,
                'min_data_in_leaf': 50,
                'min_sum_hessian_in_leaf': 5.0,
                'is_enable_sparse': True,
                'use_two_round_loading': False,
                'lambda_l1': 0,
                'lambda_l2': 0,
                'verbose': 1
            }
        }
        model_params['params'].update(params)

        evals_result = dict()
        self.model = lgb.train(train_set=self.dataset_train, valid_sets=[self.dataset_train, self.dataset_valid],
                               valid_names=['train', 'valid'], **model_params, evals_result=evals_result)
        logger.info('train summery:' + str(evals_result))
        return min(evals_result['valid']['binary_logloss'])
