import os
import argparse
import yaml
from sklearn.tree import DecisionTreeClassifier
import torch
import pickle
import numpy as np
import utils


parser = argparse.ArgumentParser("learn pddl rules from decision tree.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))

save_name = os.path.join(opts["save"], "domain.pddl")
if os.path.exists(save_name):
    os.remove(save_name)

# these are symbols from second level encoder
category = torch.load(os.path.join(opts["save"], "category.pt"))
# these are the labels from the clustering (which cluster each effect belongs to)
label = torch.load(os.path.join(opts["save"], "label.pt"))
# these are the names of the effects (aka the names of the clusters)
effect_names = np.load(os.path.join(opts["save"], "effect_names.npy"))
# number of clusters
K = len(effect_names)

tree = DecisionTreeClassifier()
# given symbols, predict the effect
tree.fit(category, label)
file = open(os.path.join(opts["save"], "tree.pkl"), "wb")
pickle.dump(tree, file)
file.close()

# symbol size
CODE_DIM = 2
#obj names are 00: objtype0, 01: objtype1, 10: objtype2, 11: objtype3
obj_names = {}
for i in range(2**CODE_DIM):
    category = utils.decimal_to_binary(i, length=CODE_DIM)
    obj_names[category] = "objtype{}".format(i)

file_loc = os.path.join(opts["save"], "domain.pddl")
if os.path.exists(file_loc):
    os.remove(file_loc)

# this line converts tree to ppdl
pddl_code = utils.tree_to_code(tree, ["f%d" % i for i in range(K)], effect_names, obj_names)
# these are extra requirements of ppdl
pretext = "(define (domain stack)\n"
pretext += "\t(:requirements :typing :negative-preconditions :probabilistic-effects :conditional-effects :disjunctive-preconditions)\n"
pretext += "\t(:predicates"

# these are the predicates
# base: the object is a base for building tower
# pickloc: the object is picked from this objects location
# instack: the object is in the stack
# stackloc: the objects stacked at this objects location
# relation0: relation between two objects
# relation1: relation between two objects
# objtype0: object type 0
# objtype1: object type 1
# objtype2: object type 2
# objtype3: object type 3
# H0: height 0
# H1: height 1
# H2: height 2
# H3: height 3
# H4: height 4
# H5: height 5
# H6: height
# S0: stack 0
# S1: stack 1
# S2: stack 2
# S3: stack 3
# S4: stack 4
# S5: stack 5
# S6: stack 6
for i in range(K):
    pretext += "\n\t\t(%s) " % effect_names[i]
pretext += "(base) \n\t\t(pickloc ?x)\n\t\t(instack ?x)\n\t\t(stackloc ?x)\n\t\t(relation0 ?x ?y)\n\t\t(relation1 ?x ?y)"
for i in range(2**CODE_DIM):
    pretext += "\n\t\t(" + obj_names[utils.decimal_to_binary(i, length=CODE_DIM)] + " ?x)"
for i in range(7):
    pretext += "\n\t\t(H%d)" % i
for i in range(7):
    pretext += "\n\t\t(S%d)" % i
pretext += "\n\t)"
print(pretext, file=open(file_loc, "a"))

# For 4x4x2 symbol probabilities define actions for each symbol
action_template = "\t(:action stack%d\n\t\t:parameters (?below ?above)"
for i, (precond, effect) in enumerate(pddl_code):
    print(action_template % i, file=open(file_loc, "a"))
    print("\t\t"+precond, file=open(file_loc, "a"))
    print("\t\t"+effect, file=open(file_loc, "a"))
    print("\t)", file=open(file_loc, "a"))
# this changes the height
# height starts from 0
# when it is stacked and height is 0 it is increased to 1
for i in range(6):
    print("\t(:action increase-height%d" % (i+1), file=open(file_loc, "a"))
    print("\t\t:precondition (and (stacked) (H%d))" % i, file=open(file_loc, "a"))
    print("\t\t:effect (and (not (H%d)) (H%d) (not (stacked)))\n\t)" % (i, i+1), file=open(file_loc, "a"))
# this changes the stack
# stack starts from 0
# when it is inserted and stack is 0 it is increased to 1
# It is important to note that stack is increased not only when the object is inserted but also when the object is stacked
# This is kind of like total number of objects used
for i in range(6):
    print("\t(:action increase-stack%d" % (i+1), file=open(file_loc, "a"))
    print("\t\t:precondition (and (inserted) (S%d))" % i, file=open(file_loc, "a"))
    print("\t\t:effect (and (not (S%d)) (S%d) (not (inserted)))\n\t)" % (i, i+1), file=open(file_loc, "a"))
print("\t(:action makebase", file=open(file_loc, "a"))
print("\t\t:parameters (?obj)", file=open(file_loc, "a"))
print("\t\t:precondition (not (base))", file=open(file_loc, "a"))
print("\t\t:effect (and (base) (stacked) (inserted) (not (pickloc ?obj)) (stackloc ?obj))", file=open(file_loc, "a"))
print("\t)", file=open(file_loc, "a"))
print(")", file=open(os.path.join(opts["save"], "domain.pddl"), "a"))
