#!/usr/bin/env python

import os
import sys
import json

from advisor_client.client import AdvisorClient
import logging

logger = logging.getLogger('xgboost grid search')

def main(train_function):
    client = AdvisorClient()

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
                "parameterName": "min_child_weight",
                "type": "INTEGER",
                "minValue": 1,
                "maxValue": 20,
                "feasiblePoints": "",
                "scallingType": "LINEAR"
            },
            {
                "parameterName": "max_depth",
                "type": "INTEGER",
                "minValue": 5,
                "maxValue": 15,
                "feasiblePoints": "",
                "scallingType": "LINEAR"
            },
            {
                "parameterName": "gamma",
                "type": "DOUBLE",
                "minValue": 0.0,
                "maxValue": 10.0,
                "feasiblePoints": "",
                "scallingType": "LINEAR"
            },
            {
                "parameterName": "alpha",
                "type": "DOUBLE",
                "minValue": 0.0,
                "maxValue": 10.0,
                "feasiblePoints": "",
                "scallingType": "LINEAR"
            },
        ]
    }
    study = client.create_study("Study", study_configuration,
                              "BayesianOptimization")
    #study = client.get_study_by_id(21)

    # Get suggested trials
    trials = client.get_suggestions(study.id, 3)

    # Generate parameters
    parameter_value_dicts = []
    for trial in trials:
        parameter_value_dict = json.loads(trial.parameter_values)
        print("The suggested parameters: {}".format(parameter_value_dict))
        parameter_value_dicts.append(parameter_value_dict)

    # Run training
    metrics = []
    for i in range(len(trials)):
        metric = train_function(**parameter_value_dicts[i])
        metrics.append(metric)

    # Complete the trial
    for i in range(len(trials)):
        trial = trials[i]
        client.complete_trial_with_one_metric(trial, metrics[i])
    is_done = client.is_study_done(study.id)
    best_trial = client.get_best_trial(study.id)
    print("The study: {}, best trial: {}".format(study, best_trial))


if __name__ == "__main__":
    curr_dir = os.path.dirname(__file__)
    logger.setLevel(logging.INFO)
    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setFormatter(logging.Formatter(fmt='{asctime}:{levelname}:{name}:{message}', style='{'))
    logger.addHandler(log_handler)

    main()
