#!/bin/bash
# These will be run only once.

# train effect prediction model
python3 train.py -opts "$1"
# cluster effect space
python3 cluster.py -opts "$1"
# save object categories
python3 save_cat.py -opts "$1"
# transform the learned prediction model into PDDL
# code by learning decision tree and encoding
# effect probabilities.
python3 learn_rules.py -opts "$1"
