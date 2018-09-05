#!/usr/bin/env python
# coding: utf-8
# pylint: disable = invalid-name, C0111

import os
import sys
import json

from advisor_client.client import AdvisorClient
import logging
try:
    from .my_model import MyModel
except ImportError:
    from models.lightgbm.my_model import MyModel

logger = logging.getLogger('lightgbm grid search')


def main():
    client = AdvisorClient()

    model = MyModel()
    model.load_dataset()
    # Get or create the study
    study_configuration = {
        "goal":
        "MINIMIZE",
        "randomInitTrials":
        5,
        "maxTrials":
        30,
        "maxParallelTrials":
        1,
        "params": [
            {
                "parameterName": "max_bin",
                "type": "INTEGER",
                "minValue": 63,
                "maxValue": 511,
                "feasiblePoints": "",
                "scallingType": "LINEAR"
            },
            {
                "parameterName": "bin_construct_sample_cnt",
                "type": "INTEGER",
                "minValue": 3,
                "maxValue": 1000,
                "feasiblePoints": "",
                "scallingType": "LINEAR"
            },
            {
                "parameterName": "num_leaves",
                "type": "INTEGER",
                "minValue": 63,
                "maxValue": 255,
                "feasiblePoints": "",
                "scallingType": "LINEAR"
            },
            # {
            #     "parameterName": "lambda_l2",
            #     "type": "DOUBLE",
            #     "minValue": 0.0,
            #     "maxValue": 10.0,
            #     "feasiblePoints": "",
            #     "scallingType": "LINEAR"
            # },
            # {
            #     "parameterName": "lambda_l1",
            #     "type": "DOUBLE",
            #     "minValue": 0.0,
            #     "maxValue": 10.0,
            #     "feasiblePoints": "",
            #     "scallingType": "LINEAR"
            # },
            {
                "parameterName": "cat_l2",
                "type": "DOUBLE",
                "minValue": 1.0,
                "maxValue": 100.0,
                "feasiblePoints": "",
                "scallingType": "LINEAR"
            },
            {
                "parameterName": "cat_smooth",
                "type": "DOUBLE",
                "minValue": 1.0,
                "maxValue": 100.0,
                "feasiblePoints": "",
                "scallingType": "LINEAR"
            },
        ]
    }
    study = client.create_study("Study", study_configuration,
                              "BayesianOptimization")
    #study = client.get_study_by_id(21)

    # Get suggested trials

    trials = client.get_suggestions(study.id, 5)
    for i in range(5):
        trial = trials[i]
        parameter_value_dict = json.loads(trial.parameter_values)
        logger.info("The suggested parameters: {}".format(parameter_value_dict))
        metric = model.train(**parameter_value_dict)
        client.complete_trial_with_one_metric(trial, metric)

    while not client.is_study_done(study.id):
        trials = client.get_suggestions(study.id, 1)
        assert len(trials) == 1
        trial = trials[0]
        parameter_value_dict = json.loads(trial.parameter_values)
        logger.info("The suggested parameters: {}".format(parameter_value_dict))
        metric = model.train(**parameter_value_dict)
        client.complete_trial_with_one_metric(trial, metric)

    best_trial = client.get_best_trial(study.id)
    logger.info("The study: {}, best trial: {}".format(study, best_trial))


if __name__ == "__main__":
    curr_dir = os.path.dirname(__file__)
    logger.setLevel(logging.INFO)
    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setFormatter(logging.Formatter(fmt='{asctime}:{levelname}:{name}:{message}', style='{'))
    logger.addHandler(log_handler)
    log_handler = logging.StreamHandler(open(os.path.join(curr_dir, 'search.log'), 'w+'))
    log_handler.setFormatter(logging.Formatter(fmt='{asctime}:{levelname}:{name}:{message}', style='{'))
    logger.addHandler(log_handler)

    main()
