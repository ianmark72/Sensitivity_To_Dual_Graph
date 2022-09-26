import facefinder
import numpy as np
import pandas as pd
import random
import pickle
import csv
import copy
import statistics
import math
import gerrychain
import networkx
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import os
import json
import argparse
from functools import partial
from gerrychain.tree import bipartition_tree as bpt
from gerrychain import Graph, MarkovChain
from gerrychain import accept
from gerrychain.constraints import (Validator, single_flip_contiguous,
                                    within_percent_of_ideal_population, UpperBound)
from gerrychain.proposals import propose_random_flip, propose_chunk_flip
from gerrychain.accept import always_accept
from gerrychain.updaters import Election, Tally, cut_edges
from gerrychain import GeographicPartition
from gerrychain.partition import Partition
from gerrychain.proposals import recom
from gerrychain.metrics import mean_median, efficiency_gap
from gerrychain.tree import recursive_tree_part, bipartition_tree_random, PopulatedGraph
from collections import defaultdict
from datetime import datetime
import time
from concurrent.futures import ProcessPoolExecutor
import logging

'''
git pull
conda activate gerry

to push:
git add {files_changed}
git commit -m "comments of changes"
git pull
git push
'''


def gerrychain_score(proposal_graph, graph, config, updaters, epsilon, ideal_population, gerrychain_steps, accept, k):



    initial_partition = Partition(proposal_graph, assignment=config['ASSIGN_COL'], updaters=updaters)


    # Sets up Markov chain
    popbound = within_percent_of_ideal_population(initial_partition, epsilon)
    tree_proposal = partial(recom, pop_col=config['POP_COL'], pop_target=ideal_population, epsilon=epsilon,
                                node_repeats=1)

    #make new function -- this computes the energy of the current map
    exp_chain = MarkovChain(tree_proposal, Validator([popbound]), accept=accept.always_accept,
                            initial_state=initial_partition, total_steps=gerrychain_steps)
    seats_won_for_republicans = []
    seats_won_for_democrats = []
    for part in exp_chain:
        rep_seats_won = 0
        dem_seats_won = 0
        for j in range(k):
            rep_votes = 0
            dem_votes = 0
            for n in graph.nodes():
                if part.assignment[n] == j:
                    rep_votes += graph.nodes[n]["EL16G_PR_R"]
                    dem_votes += graph.nodes[n]["EL16G_PR_D"]
            total_seats_dem = int(dem_votes > rep_votes)
            total_seats_rep = int(rep_votes > dem_votes)
            rep_seats_won += total_seats_rep
            dem_seats_won += total_seats_dem
        seats_won_for_republicans.append(rep_seats_won)
        seats_won_for_democrats.append(dem_seats_won)

    seat_score  = statistics.mean(seats_won_for_republicans)
    return seat_score



def test_score(proposal_graph, graph, config, updaters, epsilon, ideal_population, gerrychain_steps, accept, k):

    def vote_difference(n):
        # Energy is higher when two nodes with the same vote_difference are connected.
        # So this will prefer to connect nodes that are similar in to eachother.
        return graph.nodes[n]["EL16G_PR_R"] - graph.nodes[n]["EL16G_PR_D"]
    def democrat_votes_only(n):
        # Energy is higher when two nodes with that have a high proportion of democrat votes  are connected.
        # So this will prefer to connect nodes that are in densely blue areas.
        return graph.nodes[n]["EL16G_PR_D"] / ( graph.nodes[n]["EL16G_PR_R"] + graph.nodes[n]["EL16G_PR_D"])

    displacement_energy = 0
    for n in graph.nodes():
        graph.nodes[n]["charge"] = vote_difference(n)
    for e in graph.edges():
        displacement_energy += graph.nodes[e[0]]["charge"] *  graph.nodes[e[1]]["charge"]
        
    print("energy", displacement_energy)
    return displacement_energy
