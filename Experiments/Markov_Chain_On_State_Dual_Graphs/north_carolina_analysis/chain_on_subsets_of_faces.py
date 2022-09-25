## This script will perform a markov chain on the  subset faces of the north carolina graph, picking a face UAR and sierpinskifying or de-sierpinskifying it, then running gerrychain on the graph, recording central seat tendencies.
# the output of the chain is stored in north_carolina/plots in a pickled object.
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

from metachain_scores import *

def face_sierpinski_mesh(graph, special_faces):
    """'Sierpinskifies' certain faces of the graph by adding nodes and edges to
    certain faces.

    Args:
        partition (Gerrychain Partition): partition object which contain assignment
        and whose graph will have edges and nodes added to
        special_faces (list): list of faces that we want to add node/edges to

    Raises:
        RuntimeError if SIERPINSKI_POP_STYLE of config file is neither 'uniform'
        nor 'zero'
    """

    # Get maximum node label.
    label = max(list(graph.nodes()))
    # Assign each node to its district in partition
    #for node in graph.nodes():
    #    graph.nodes[node][config['ASSIGN_COL']] = partition.assignment[node]

    for face in special_faces:
        neighbors = [] #  Neighbors of face
        locationCount = np.array([0,0]).astype("float64")
        # For each face, add to neighbor_list and add to location count
        for vertex in face:
            neighbors.append(vertex)
            locationCount += np.array(graph.nodes[vertex]["pos"]).astype("float64")
        # Save the average of each of the face's positions
        facePosition = locationCount / len(face)

        # In order, store relative position of each vertex to the position of the face
        locations = [graph.nodes[vertex]['pos'] - facePosition for vertex in face]
        # Sort neighbors according to each node's angle with the center of the face
        angles = [float(np.arctan2(x[0], x[1])) for x in locations]
        neighbors.sort(key=dict(zip(neighbors, angles)).get)

        newNodes = []
        newEdges = []
        # For each consecutive pair of nodes, remove their edge, create a new
        # node at their average position, and connect edge node to the new node:
        for vertex, next_vertex in zip(neighbors, neighbors[1:] + [neighbors[0]]):
            label += 1
            # Add new node to graph with corresponding label at the average position
            # of vertex and next_vertex, and with 0 population and 0 votes
            graph.add_node(label)
            avgPos = (np.array(graph.nodes[vertex]['pos']) +
                      np.array(graph.nodes[next_vertex]['pos'])) / 2
            graph.nodes[label]['pos'] = avgPos
            graph.nodes[label][config['X_POSITION']] = avgPos[0]
            graph.nodes[label][config['Y_POSITION']] = avgPos[1]
            graph.nodes[label][config['POP_COL']] = 0
            graph.nodes[label][config['PARTY_A_COL']] = 0
            graph.nodes[label][config['PARTY_B_COL']] = 0

            # For each new node, 'move' a third of the population, Party A votes,
            # and Party B votes from its two adjacent nodes which previously exists
            # to itself (so that each previously existing node equally shares
            # its statistics with the two new nodes adjacent to it)
            if config['SIERPINSKI_POP_STYLE'] == 'uniform':
                for vert in [vertex, next_vertex]:
                    for keyword, orig_keyword in zip(['POP_COL', 'PARTY_A_COL', 'PARTY_B_COL'],
                                                     ['orig_pop', 'orig_A', 'orig_B']):
                        # Save original values if not done already
                        if orig_keyword not in graph.nodes[vert]:
                            graph.nodes[vert][orig_keyword] = graph.nodes[vert][config[keyword]]

                        # Increment values of new node and decrement values of old nodes
                        # by the appropriate amount.
                        graph.nodes[label][config[keyword]] += graph.nodes[vert][orig_keyword] // 3
                        graph.nodes[vert][config[keyword]] -= graph.nodes[vert][orig_keyword] // 3

                # Assign new node to same district as neighbor. Note that intended
                # behavior is that special_faces do not correspond to cut edges,
                # and therefore both vertex and next_vertex will be of the same
                # district.
                graph.nodes[label][config['ASSIGN_COL']] = graph.nodes[vertex][config['ASSIGN_COL']]
             # Choose a random adjacent node, assign the new node to the same partition,
            # and move half of its votes and population to said node
            elif config['SIERPINSKI_POP_STYLE'] == 'random':
                chosenNode = random.choice([graph.nodes[vertex], graph.nodes[next_vertex]])
                graph.nodes[label][config['ASSIGN_COL']] = chosenNode[config['ASSIGN_COL']]
                for keyword in ['POP_COL', 'PARTY_A_COL', 'PARTY_B_COL']:
                    graph.nodes[label][config[keyword]] += chosenNode[config[keyword]] // 2
                    chosenNode[config[keyword]] -= chosenNode[config[keyword]] // 2
            # Set the population and votes of the new nodes to zero. Do not change
            # previously existing nodes. Assign to random neighbor.
            elif config['SIERPINSKI_POP_STYLE'] == 'zero':
                graph.nodes[label][config['ASSIGN_COL']] =\
                random.choice([graph.nodes[vertex][config['ASSIGN_COL']],
                               graph.nodes[next_vertex][config['ASSIGN_COL']]]
                             )
            else:
                raise RuntimeError('SIERPINSKI_POP_STYLE must be "uniform" or "zero"')

            # Remove edge between consecutive nodes if it exists
            if graph.has_edge(vertex, next_vertex):
                graph.remove_edge(vertex, next_vertex)

            # Add edge between both of the original nodes and the new node
            graph.add_edge(vertex, label)
            newEdges.append((vertex, label))
            graph.add_edge(label, next_vertex)
            newEdges.append((label, next_vertex))
            # Add node to connections
            newNodes.append(label)

        # Add an edge between each consecutive new node
        for vertex in range(len(newNodes)):
            graph.add_edge(newNodes[vertex], newNodes[(vertex+1) % len(newNodes)])
            newEdges.append((newNodes[vertex], newNodes[(vertex+1) % len(newNodes)]))
        # For each new edge of the face, set sibilings to be the tuple of all
        # new edges
        siblings = tuple(newEdges)
        for edge in newEdges:
            graph.edges[edge]['siblings'] = siblings
def add_edge_proposal(graph, special_faces):
    """Takes set of 4 edge faces, and adds an edge.

    Args:
        graph (Gerrychain Graph): graph in JSON file following cleaning
        special_faces (List): list of four sided faces
    """
    for face in special_faces:
        for vertex in face:
            for itr_vertex in face:
                if ((not graph.has_edge(vertex, itr_vertex)) and (not graph.has_edge(itr_vertex, vertex)) and vertex != vertex):
                    graph.add_edge(vertex, itr_vertex)
                    graph.edges[ (vertex, itr_vertex)] .new = True # For drawing it with a different color later. Todo: check my syntax.
                    break


def preprocessing(path_to_json, output_directory):
    """Takes file path to JSON graph, and returns the appropriate

    Args:
        path_to_json ([String]): path to graph in JSON format
        output_directory: path of directory to save output in
    Returns:
        graph (Gerrychain Graph): graph in JSON file following cleaning
        dual (Gerrychain Graph): planar dual of graph
    """
    graph = Graph.from_json(path_to_json)
    # For each node in graph, set 'pos' keyword to position
    for node in graph.nodes():
        graph.nodes[node]['pos'] = (graph.nodes[node][config['X_POSITION']],
                                    graph.nodes[node][config['Y_POSITION']])

    save_fig(graph, output_directory + '/UnderlyingGraph.png', config['WIDTH'])
    #restricted planar dual does not include unbounded face
    dual = facefinder.restricted_planar_dual(graph)

    return graph, dual

def validate_map(seat_score, popbound, k, proposal_graph, special_faces, base_score, chain_output, tree_proposal, initial_partition, graph, step, thread_directory):
    output_directory = thread_directory
    exp_chain = MarkovChain(tree_proposal, Validator([popbound]), accept=accept.always_accept,
                        initial_state=initial_partition, total_steps=config['BASELINE_STEPS'])
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
    validated_seat_score  = statistics.mean(seats_won_for_republicans)
    # add .1 to ensure substantial
    #print("Found Validated Score of: ", validated_seat_score, " at step: ", i,)
    with open(output_directory + '/rep_seats.json', 'w', encoding='utf-8') as f:
        json.dump(seats_won_for_republicans, f, ensure_ascii=False, indent=4)
    with open(output_directory + '/dem_seats.json', 'w', encoding='utf-8') as f:
        json.dump(seats_won_for_democrats, f, ensure_ascii=False, indent=4)
    nx.write_gpickle(proposal_graph, output_directory + '/' + "_highest_score_graph", pickle.HIGHEST_PROTOCOL)
    f= open(output_directory + "/max_score_data.txt","w+")
    f.write("Validated Mean Seats for Republicans: " + str(validated_seat_score) + "\n" + "Edges changed: " + str(len(special_faces)) + "\n" + "Pre-validation Seats: " + str(seat_score) + '\n' "Baseline Seats: " + str(base_score) + '\n' + "Step " + str(step) +'\n' + "Experiment start: " + config["EXPERIMENT_START"])
    f.close()
    save_obj(special_faces, output_directory + '/', "special_faces")
    try:
        plt.plot(range(len(chain_output['score'])), chain_output['score'])
        plt.xlabel("Meta-Chain Step")
        plt.ylabel("Score")
        plot_name = output_directory + '/' + 'score'+ '.png'
        plt.savefig(plot_name)
        plt.close()
        plt.plot(range(len(chain_output['acceptance_probability'])), chain_output['acceptance_probability'])
        plt.xlabel("Meta-Chain Step")
        plt.ylabel("Acceptance Probability")
        plot_name = output_directory + '/' + 'max_score_probability'+ '.png'
        plt.savefig(plot_name)
    except:
        print("Error plotting")

    #logging.info("Thread %s: completed", step)


def save_fig(graph, path, size):
    """Saves graph to file in desired formed

    Args:
        graph (Gerrychain Graph): graph to be saved
        path (String): path to file location
        size (int): width of image
    """
    plt.figure()
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size=1    , width=size, cmap=plt.get_cmap('jet'))
    # Gets format from end of filename
    plt.savefig(path, format=path.split('.')[-1])
    plt.close()
def test_process(seat_score, popbound, k, proposal_graph, special_faces, base_score, chain_output, tree_proposal, initial_partition, graph, step,thread_directory):
    output_directory = thread_directory
    print("Validation    : Entered process, output dir: %s", output_directory)
    with open(output_directory + "/max_score_data.txt","w+") as f:
        f.write("Validated Mean Seats for Republicans: " + str(8) + "\n" + "Edges changed: " + str(77) + "\n" + "Pre-validation Seats: " + str(9) + '\n' "Baseline Seats: " + str(9) + '\n' + "Step " + str(100) +'\n' + "Experiment start: " + config["EXPERIMENT_START"])
        f.flush()


def main():
    """ Contains majority of expermiment. Runs a markov chain on the state dual graph, determining how the distribution is affected to changes in the
     state dual graph.
     Raises:
        RuntimeError if PROPOSAL_TYPE of config file is neither 'sierpinski'
        nor 'convex'
    """
    output_directory = createDirectory(config)
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")
    logging.info("Main    : Configurations: %s", config)
    epsilon = config["METACHAIN_EPSILON"]
    k = config["NUM_DISTRICTS"]
    updaters = {'population': Tally('population'),
                            'cut_edges': cut_edges,
                            }
    graph, dual = preprocessing(config["INPUT_GRAPH_FILENAME"], output_directory)
    ideal_population= sum( graph.nodes[x]["population"] for x in graph.nodes())/k
    faces = graph.graph["faces"]
    faces = list(faces)
    square_faces = [face for face in faces if len(face) == 4]
    totpop = 0
    for node in graph.nodes():
        totpop += int(graph.nodes[node]['population'])
    #length of chain
    steps = config["CHAIN_STEPS"]

    #length of each gerrychain step
    gerrychain_steps = config["GERRYCHAIN_STEPS"]
    #faces that are currently modified. Code maintains list of modified faces, and at each step selects a face. if face is already in list,
    #the face is un-modified, and if it is not, the face is modified by the specified proposal type.
    #compute baseline map score
    initial_partition = Partition(graph, assignment=config['ASSIGN_COL'], updaters=updaters)


    # Sets up Markov chain
    popbound = within_percent_of_ideal_population(initial_partition, epsilon)
    tree_proposal = partial(recom, pop_col=config['POP_COL'], pop_target=ideal_population, epsilon=epsilon,
                                node_repeats=1)

    logging.info("Main    : Computing Baseline Seats")
    #Calculate baseline score to get a sense of the typical distribution
    # exp_chain = MarkovChain(tree_proposal, Validator([popbound]), accept=accept.always_accept,
    #                         initial_state=initial_partition, total_steps=config['BASELINE_STEPS'])
    # seats_won_for_republicans = []
    # seats_won_for_democrats = []
    # for part in exp_chain:
    #     rep_seats_won = 0
    #     dem_seats_won = 0
    #     for j in range(k):
    #         rep_votes = 0
    #         dem_votes = 0
    #         for n in graph.nodes():
    #             if part.assignment[n] == j:
    #                 rep_votes += graph.nodes[n]["EL16G_PR_R"]
    #                 dem_votes += graph.nodes[n]["EL16G_PR_D"]
    #         total_seats_dem = int(dem_votes > rep_votes)
    #         total_seats_rep = int(rep_votes > dem_votes)
    #         rep_seats_won += total_seats_rep
    #         dem_seats_won += total_seats_dem
    #     seats_won_for_republicans.append(rep_seats_won)
    #     seats_won_for_democrats.append(dem_seats_won)

    #base_score  = statistics.mean(seats_won_for_republicans)
    base_score = config['BASE_SCORE']
    threshold_to_beat = base_score + .1
    logging.info("Main    : Baseline Seats: %s", base_score)
    special_faces = set( [ face for face in square_faces if np.random.uniform(0,1) < .5 ] )
    chain_output = defaultdict(list)
    #start with small score to move in right direction
    max_score = -math.inf
    #this is the meta-chain, which modifies the graph and measures the distribution change
    tmp_ctr = 0
    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(1,steps+1):
            special_faces_proposal = copy.deepcopy(special_faces)
            proposal_graph = copy.deepcopy(graph)
            if (config["PROPOSAL_TYPE"] == "sierpinski"):
                for i in range(math.floor(len(faces) * config['PERCENT_FACES'])):
                    face = random.choice(faces)
                    ##Makes the Markov chain lazy -- this just makes the chain aperiodic.
                    if random.random() > .5:
                        if not (face in special_faces_proposal):
                            special_faces_proposal.append(face)
                        else:
                            special_faces_proposal.remove(face)
                face_sierpinski_mesh(proposal_graph, special_faces_proposal)
            elif(config["PROPOSAL_TYPE"] == "add_edge"):
                change_ctr = 0
                for j in range(math.floor(len(square_faces) * config['PERCENT_FACES'])):
                    face = random.choice(square_faces)
                    ##Makes the Markov chain lazy -- this just makes the chain aperiodic.
                    if random.random() > .5:
                        if not (face in special_faces_proposal):
                            change_ctr += 1
                            special_faces_proposal.add(face)
                        else:
                            special_faces_proposal.remove(face)
                add_edge_proposal(proposal_graph, special_faces_proposal)
            else:
                raise RuntimeError('PROPOSAL TYPE must be "sierpinski" or "convex"')


            if config['metachain_score'] == "gerrychain_score":
                seat_score = gerrychain_score(proposal_graph, graph, config, updaters, epsilon, ideal_population, gerrychain_steps, accept)

            if config['metachain_score'] == "test":
                seat_score = test_score(proposal_graph, graph, config, updaters, epsilon, ideal_population, gerrychain_steps, accept)



            # If the map we have found beats the baseline map, run gerrychain for longer to see distribution
            if seat_score > threshold_to_beat:
                logging.info("Main    : Found potential map of score: %s Current Threshold: %s starting validation thread %s", seat_score, threshold_to_beat, i)
                #print("Found potential map of score: ", seat_score, " Current Threshold: ", threshold_to_beat, "Starting validation")
                thread_directory = createDirectory(config)
                # futures.append(executor.submit(test_process, int(seat_score), within_percent_of_ideal_population(initial_partition, config['VALIDATION_EPSILON']), int(k), copy.deepcopy(proposal_graph), copy.deepcopy(special_faces), float(base_score), copy.deepcopy(chain_output), partial(recom, pop_col=config['POP_COL'], pop_target=ideal_population, epsilon=config['VALIDATION_EPSILON'],
                #                         node_repeats=1), initial_partition, copy.deepcopy(graph), int(i),thread_directory))
                validate_map(int(seat_score), within_percent_of_ideal_population(initial_partition, config['VALIDATION_EPSILON']), int(k), copy.deepcopy(proposal_graph), copy.deepcopy(special_faces), float(base_score), copy.deepcopy(chain_output), partial(recom, pop_col=config['POP_COL'], pop_target=ideal_population, epsilon=config['VALIDATION_EPSILON'],
                                        node_repeats=1), initial_partition, copy.deepcopy(graph), int(i),thread_directory)
                # x = Thread(target=validate_map, args=(int(seat_score), within_percent_of_ideal_population(initial_partition, config['VALIDATION_EPSILON']), int(k), copy.deepcopy(proposal_graph), copy.deepcopy(special_faces), float(base_score), copy.deepcopy(chain_output), partial(recom, pop_col=config['POP_COL'], pop_target=ideal_population, epsilon=config['VALIDATION_EPSILON'],
                #                         node_repeats=1), initial_partition, copy.deepcopy(graph), int(i)))
                #x.start()
                logging.info("Main    : Validation thread %s started", i )
                #validate_map(int(seat_score), within_percent_of_ideal_population(initial_partition, epsilon), int(k), copy.deepcopy(proposal_graph), copy.deepcopy(special_faces), int(base_score),copy.deepcopy(chain_output), partial(recom, pop_col=config['POP_COL'], pop_target=ideal_population, epsilon=epsilon,
                                        #node_repeats=1), initial_partition, copy.deepcopy(graph), int(i))


            #implement modified mattingly simulated annealing scheme, from evaluating partisan gerrymandering in wisconsin
            # if i <= math.floor(steps * .67):
            #     beta = i / math.floor(steps * .67)
            # else:
            #     beta = (i / math.floor(steps * .67)) * 100
            # temperature = 1 / (beta)
            #set up basic temperature cooling, linear decrease
            #temperature function
            #\ 50\left(\exp\left(\operatorname{mod}\left(x,1500\right)\ \cdot\ -\frac{5}{1500}\right)-\ \exp\left(-5\right)\right)
            if (i == 50) and (score + .1 < (threshold_to_beat - .1)):
                logging.info("Main    : resetting temp counter, score %s failed to pass threshold: %s starting validation thread %s", score, (threshold_to_beat - .1))
            # print("resetting temp counter, score ", score, "failed to pass threshold", (threshold_to_beat - .1))
                tmp_ctr = 0
            temperature =  50 * ((math.exp( (tmp_ctr % 1500) * -(5/1500))) - math.exp(-5))
            tmp_ctr += 1
            #acceptance probability
            # y\ =\ (\exp(s)/\exp(l))^{\left(1/x)\right)}\
            # weight_seats = config['WEIGHT_SEATS'] = 1
            # weight_flips = config['WEIGHT_FLIPS'] = 1
            # flip_score = len(special_faces)
            # This is the number of edges being swapped
            #score = weight_seats * seat_score + weight_flips *  flip_score
            score = seat_score
            if i != 1:
                acceptance_criteria = (math.exp(score) / math.exp(chain_output['score'][-1]))**(1/temperature)
            else:
                acceptance_criteria = 1
            logging.info("Main    : Step , %s , acceptance probability : %s", i, acceptance_criteria)
            #print ("step ", i , "acceptance probability ", acceptance_criteria)
            ##This is the acceptance step of the Metropolis-Hasting's algorithm. Specifically, rand < min(1, P(x')/P(x)), where P is the energy and x' is proposed state
            #if the acceptance criteria is met or if it is the first step of the chain

            def accept_state():
                """ Accept the next state of the meta-chain and update the chain output with the proposed changes.
                    To track any new information during the chain add a key to the dictionary and append the desired information.
                    """
                #chain_output['dem_seat_data'].append(seats_won_for_democrats)
                #chain_output['rep_seat_data'].append(seats_won_for_republicans)
                chain_output['score'].append(score)
                #chain_output['seat_score'].append(seat_score)
                chain_output['acceptance_probability'].append(acceptance_criteria)
                #chain_output['flip_score'].append(flip_score)
            def reject_state():
                """ Reject the next state of the meta-chain and propogate the current state of the meta-chain
                    """
                for key in chain_output.keys():
                    chain_output[key].append(chain_output[key][-1])
            #logging.info("Main    :Chain probability so far: , %s", chain_output["acceptance_probability"])
            #print("Chain prbability so far: ", chain_output["acceptance_probability"] )
            #this is the simplified form of the acceptance criteria, for intuitive purposes
            #exp((1/temperature) ( proposal_score - previous_score))
            if np.random.uniform(0,1) < acceptance_criteria:
                #accept changes
                accept_state()
                special_faces = copy.deepcopy(special_faces_proposal)
            else:
                #reject changes
                reject_state()
            if i % 1500 == 0:
                plt.plot(range(len(chain_output['acceptance_probability'])), chain_output['acceptance_probability'])
                plt.xlabel("Meta-Chain Step")
                plt.ylabel("Acceptance Probability")
                plot_name = output_directory + '/' + 'overall_chain_probability'+ '.png'
                plt.savefig(plot_name)
   #save_obj(chain_output, output_directory, "chain_output")
def createDirectory(config):
    """Creates experiment directory to track experiment configuration information and output.
    Written by Matt Karramann.
    Args:
        config file for experiment
    """
    num = 0
    suffix = lambda x: f'/{x}' if x != 0 else ''
    while os.path.exists(config['EXPERIMENT_NAME'] + suffix(num)):
        num += 1
    os.makedirs(config['EXPERIMENT_NAME'] + suffix(num))
    metadataFile = os.path.join(config['EXPERIMENT_NAME'] + suffix(num), config['METADATA_FILE'])
    with open(metadataFile, 'w') as metaFile:
        json.dump(config, metaFile, indent=2)
    return config['EXPERIMENT_NAME'] + suffix(num)

def save_obj(obj, output_directory, name):
    with open(output_directory + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
if __name__ ==  '__main__':
    # parser = argparse.ArgumentParser(description='Runs a metachain on the space of state dual graphs.')
    # parser.add_argument("-mcs", "--meta_chain_steps", help="Length of metachain", type=int)
    # parser.add_argument("-gcs", "--gerry_chain_steps", help="Length of gerrychain steps", type=int)
    # parser.add_argument("-ws", "--weight_seats", help="score weight of seats achieved", type=float)
    # parser.add_argument("-wf", "--weight_flips", help="Score weight of flips", type=float)
    # args = parser.parse_args()
    global config
    config = {
        "INPUT_GRAPH_FILENAME" : "./jsons/NC.json",
        "X_POSITION" : "C_X",
        "Y_POSITION" : "C_Y",
        'PARTY_A_COL': "EL16G_PR_R",
        'PARTY_B_COL': "EL16G_PR_D",
        "UNDERLYING_GRAPH_FILE" : "./plots/UnderlyingGraph.png",
        "WIDTH" : 1,
        "ASSIGN_COL" : "part",
        "POP_COL" : "population",
        'SIERPINSKI_POP_STYLE': 'random',
        'GERRYCHAIN_STEPS' : 15,
        'CHAIN_STEPS' : 4500,
        'BASELINE_STEPS': 10,
        "NUM_DISTRICTS": 13,
        'STATE_NAME': 'north_carolina',
        'PERCENT_FACES': 1,
        'PROPOSAL_TYPE': "add_edge",
        'METACHAIN_EPSILON': .1,
        'VALIDATION_EPSILON': .05,
        "EXPERIMENT_NAME" : 'experiments/north_carolina/' + str(datetime.now()),
        'METADATA_FILE' : "experiment_data",
        'WEIGHT_SEATS' : 0,
        'WEIGHT_FLIPS' : 0,
        'EXPERIMENT_START': str(datetime.now()),
        'BASE_SCORE': 7.75,
        'metachain_score' : "gerrychain_score"
    }
    # Seanna: so in here the number of districts is 12 (maybe we want to revise it?)
    main()

    '''
    weight_pairs = np.random.normal(0,1, (100,2))
    # Scatterplot of the seatshift, edges flipped
    for seat shift, save both median and mean , and 5% and 95% quantiles
    # Sense of scale, e.g. by adding a point to the scatter plot representing the original state dual graph.
    '''
