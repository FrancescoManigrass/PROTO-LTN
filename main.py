#!/usr/bin/env python
# coding: utf-8

# In[1]:
import argparse
import json
import sys
import tensorflow as tf
from config import config
from metrics import metrics_dict_history, scheduled_parameters
from neptune_class import neptune_class
from train import train

config_file = config()
neptune_experiemnt = neptune_class(config_file)




def main(args):
    if args.learning_rate:
        config_file.learning_rate=float(args.learning_rate)
    if args.alpha:
        config_file.alpha=float(args.alpha)
    if args.regularization_parameter:
        config_file.regularization_parameter=float(args.regularization_parameter)
    if args.loss:
        config_file.loss = args.loss
    if args.forAllAggregator:
        config_file.forAllAggregator=args.forAllAggregator
    if args.experiment_name:
        config_file.experiment_name=args.experiment_name
    if args.negation_axioms:
        config_file.negation_axioms = json.loads(args.negation_axioms.lower())
    if args.similarity:
        config_file.similarity = args.similarity
    if args.epochs:
        config_file.epochs = int(args.epochs)
    if args.pretrained:
        config_file.pretrained = json.loads(args.pretrained.lower())
    if args.weights:
        config_file.weights = args.weights


    config_file.neptune_experiemnt=neptune_class(config_file)
    if config_file.neptune_flag:
        config_file.neptune_experiemnt.create_experiment_neptune()
    lr=config_file.learning_rate



    for metrics in metrics_dict_history.values():
        metrics=[]

    train(
        config_file

    )

    if config_file.neptune_flag:
        for metrics in metrics_dict_history.keys():
            config_file.neptune_experiemnt.log_history_metric(metrics, metrics_dict_history[metrics])

        config_file.neptune_experiemnt.stop()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='My script')
    parser.add_argument('--learning_rate', help="Subfolder of C:\TEMP\ to manipulate",default=None)
    parser.add_argument('--alpha', help="Subfolder of C:\TEMP\ to manipulate", default=None)
    parser.add_argument('--regularization_parameter', help="Subfolder of C:\TEMP\ to manipulate", default=None)
    parser.add_argument('--negation_axioms', help="Subfolder of C:\TEMP\ to manipulate", default=None)
    parser.add_argument('--loss', help="Subfolder of C:\TEMP\ to manipulate", default=None)
    parser.add_argument('--forAllAggregator', help="Subfolder of C:\TEMP\ to manipulate", default=None)
    parser.add_argument('--experiment_name', help="Subfolder of C:\TEMP\ to manipulate", default=None)
    parser.add_argument('--similarity', help="Subfolder of C:\TEMP\ to manipulate", default=None)
    parser.add_argument('--epochs', help="Subfolder of C:\TEMP\ to manipulate", default=None)
    parser.add_argument('--pretrained', help="Subfolder of C:\TEMP\ to manipulate", default=None)
    parser.add_argument('--weights', help="Subfolder of C:\TEMP\ to manipulate", default=None)
    #parser.add_argument('--dropout', help="Subfolder of C:\TEMP\ to manipulate", default=None)

    args = parser.parse_args(sys.argv[1:])
    main(args)
