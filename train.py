"""
In this file, functions are used in training process.
And function worked on both networks.
Functions worked on the model itself is implemented as model attr
"""
import torch
import torch.nn as nn

from FINDER import FINDER
from functional import Functional

import numpy as np
import networkx as nx
import random
import time
import pickle as cp
import sys
from tqdm import tqdm
import PrepareBatchGraph  # C++
import graph  # C++
import nstep_replay_mem  # C++
import nstep_replay_mem_prioritized  # C++
import mvc_env  # C++
import utils  # C++
import scipy.linalg as linalg
import os

# region global vars

# # hyper param
NUM_MIN = 30
NUM_MAX = 50
MAX_ITERATION = 500000
BATCH_SIZE = 64
REG_HIDDEN = 32
UPDATE_TIME = 1000
MEMORY_SIZE = 500000
LEARNING_RATE = 0.0001
GAMMA = 1
N_STEP = 5

eps_start = 1.0
eps_end = 0.05
eps_step = 10000.0
n_valid = 200
aux_dim = 4
num_env = 1
inf = 2147483647 / 2

# # reinforcement learning
g_type = 'barabasi_albert'
TrainSet = graph.py_GSet()
TestSet = graph.py_GSet()
nStepReplayMem = nstep_replay_mem.py_NStepReplayMem(MEMORY_SIZE)
env_list = []
g_list = []

for i in range(num_env):
    env_list.append(mvc_env.py_MvcEnv(NUM_MAX))
    g_list.append(graph.py_Graph())
test_env = mvc_env.py_MvcEnv(NUM_MAX)
ngraph_test = 0
ngraph_train = 0
# # device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # utils
F = Functional()


# endregion

# region rewrite func
def prepare_validation_data():
    print('\ngenerating validation graphs...')
    sys.stdout.flush()

    result_degree = 0.0
    result_betweeness = 0.0

    for t in tqdm(range(n_valid)):
        g = gen_graph(NUM_MIN, NUM_MAX)
        g_degree = g.copy()
        g_betweenness = g.copy()
        val_degree, sol = F.HXA(g_degree, 'HDA')
        result_degree += val_degree
        val_betweenness, sol = F.HXA(g_betweenness, 'HBA')
        result_betweeness += val_betweenness
        insert_graph(g, is_test=True)
    print('Validation of HDA: %.6f' % (result_degree / n_valid))
    print('Validation of HBA: %.6f' % (result_betweeness / n_valid))


def gen_new_graphs(num_min, num_max):
    print('\ngenerating new training graphs...')
    sys.stdout.flush()
    clear_train_graphs()
    
    for i in tqdm(range(1000)):
        g = gen_graph(num_min, num_max)
        insert_graph(g, is_test=False)

def gen_graph(num_min, num_max):
    cur_n = np.random.randint(num_max-num_min+1) + num_min
    if g_type == 'erdos_renyi':
        g = nx.erdos_renyi_graph(n=cur_n, p=0.15)
    elif g_type == 'powerlaw':
        g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
    elif g_type == 'small-world':
        g = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
    elif g_type == 'barabasi_albert':
        g = nx.barabasi_albert_graph(n=cur_n, m=4)
    else:
        raise ValueError("g is not given value")
    return g

def insert_graph(g, is_test):
    global ngraph_test, ngraph_train
    if is_test:
        t = ngraph_test
        ngraph_test += 1
        TestSet.InsertGraph(t, F.GenNetwork(g))
    else:
        t = ngraph_train
        ngraph_train += 1
        TrainSet.InsertGraph(t, F.GenNetwork(g))

def take_snapshot(network: nn.Module, target_net: nn.Module):
    target_net.load_state_dict(network.state_dict())


def play_game(n_traj, epsilon):
    run_simulator(n_traj, epsilon, TrainSet, N_STEP)


def run_simulator(num_seq, epsilon, train_set, n_step):
    global num_env

    n = 0
    while n < num_seq:
        for i in range(num_env):
            if env_list[i].graph.num_nodes == 0 or env_list[i].isTerminal():
                if env_list[i].graph.num_nodes > 0 and env_list[i].isTerminal():
                    n = n + 1
                    nStepReplayMem.Add(env_list[i], n_step)

                g_sample = train_set.Sample()
                env_list[i].s0(g_sample)
                g_list[i] = env_list[i].graph

        if n >= num_seq:
            break

        randomly = False
        if random.uniform(0, 1) >= epsilon:
            # predict q_on_all using current (not target) network
            pred = predict_with_current_q_net(g_list, [env.action_list for env in env_list])
        else:
            randomly = True

        # step on this predict result
        # a_t is the chosen node in the subgraph
        for i in range(num_env):
            if randomly:
                a_t = env_list[i].randomAction()
            else:
                a_t = F.argMax(pred[i])
            env_list[i].step(a_t)

def predict_with_current_q_net(graph_list, covered):
    return current_network.predict(graph_list, covered)

def predict_with_snapshot(sample_g_list, covered):
    return target_network.predict(sample_g_list, covered)

def test(gid):
    graph_list = []
    test_env.s0(TestSet.Get(gid))
    graph_list.append(test_env.graph)

    cost = 0
    sol=[]

    while not test_env.isTerminal():

        list_pred = predict_with_current_q_net(graph_list, [test_env.action_list])
        new_action = F.argMax(list_pred[0])
        test_env.stepWithoutReward(new_action)
        sol.append(new_action)
    nodes = list(range(graph_list[0].num_nodes))
    solution = sol + list(set(nodes) ^ set(sol))
    robustness = F.utils.getRobustness(graph_list[0], solution)

    return robustness

def save_model():
    torch.save(current_network, model_path)
    torch.save(target_network, model_path)

def fit():
    sample = nStepReplayMem.Sampling(BATCH_SIZE)
    ness = False

    for f in range(BATCH_SIZE):
        if not sample.list_term[f]:
            ness = True
            break
    if ness:
        list_pred = predict_with_snapshot(sample.g_list, sample.list_s_primes)

    list_target = np.zeros([BATCH_SIZE, 1])
    for f in range(BATCH_SIZE):
        q_rhs = 0
        if not sample.list_term[f]:
            q_rhs = GAMMA * F.Max(list_pred[f])
        q_rhs += sample.list_rt[f]
        list_target[f] = q_rhs

    # print("size of target in fit(): ", list_target.shape)
    # print("size of batch: ", BATCH_SIZE)

    loss = current_network.fit(sample.g_list, sample.list_st, sample.list_at, list_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def clear_train_graphs():
    global ngraph_train
    ngraph_train = 0
    TrainSet.Clear()
# endregion


# training vars
current_network = FINDER()
target_network = FINDER()
current_network.cuda()
target_network.cuda()
optimizer = torch.optim.Adam(current_network.parameters(), lr=LEARNING_RATE)

N_start = None
N_end = None

# prepare validation data
prepare_validation_data()

# initialize experience buffer
gen_new_graphs(NUM_MIN, NUM_MAX)
for i in range(10):
    play_game(100, 1)
take_snapshot(current_network, target_network)

# file write
save_dir = './models/Model_powerlaw'
VCFile = '%s/ModelVC_%d_%d.csv' % (save_dir, NUM_MIN, NUM_MAX)


if not os.path.exists(save_dir):
    os.mkdir(save_dir)
f_out = open(VCFile, 'w')
# Training
for iteration in range(MAX_ITERATION):
    print(f"\rtraining iteration: {iteration}/{MAX_ITERATION}", end="", flush=True)
    start = time.time()

    if iteration and iteration % 5000 == 0:
        gen_new_graphs(NUM_MIN, NUM_MAX)
    eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iteration) / eps_step)

    if iteration % 10 == 0:
        play_game(10, eps)

    if iteration % 300 == 0:
        if iteration == 0:
            N_start = start
        else:
            N_start = N_end
        frac = 0.0
        test_start = time.time()
        for idx in range(n_valid):
            frac += test(idx)
        test_end = time.time()
        f_out.write('%.16f\n' % (frac / n_valid))  # write vc into the file
        f_out.flush()
        print('\niter %d, eps %.4f, average size of vc:%.6f' % (iteration, eps, frac / n_valid))
        print('testing 200 graphs time: %.2fs' % (test_end - test_start))
        N_end = time.time()
        print('300 iterations total time: %.2fs\n' % (N_end - N_start))
        sys.stdout.flush()
        model_path = '%s/nrange_%d_%d_iter_%d.ckpt' % (save_dir, NUM_MIN, NUM_MAX, iteration)
        save_model()

    if iteration % UPDATE_TIME == 0:
        take_snapshot(current_network, target_network)

    fit()



