import sys

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn.functional import relu, normalize, huber_loss, mse_loss
from torch import sparse

import numpy as np

import PrepareBatchGraph
from functional import truncated_normal

# Hyper Parameters:
cdef double GAMMA = 1  # decay rate of past observations
cdef int UPDATE_TIME = 1000
cdef int EMBEDDING_SIZE = 64
cdef int MAX_ITERATION = 500000
cdef double LEARNING_RATE = 0.0001  #dai
cdef int MEMORY_SIZE = 500000
cdef double Alpha = 0.001  ## weight of reconstruction loss
########################### hyperparameters for priority(start)#########################################
cdef double epsilon = 0.0000001  # small amount to avoid zero priority
cdef double alpha = 0.6  # [0~1] convert the importance of TD error to priority
cdef double beta = 0.4  # importance-sampling, from initial value increasing to 1
cdef double beta_increment_per_sampling = 0.001
cdef double TD_err_upper = 1.  # clipped abs error
########################## hyperparameters for priority(end)#########################################
cdef int N_STEP = 5
cdef int NUM_MIN = 30
cdef int NUM_MAX = 50
cdef int REG_HIDDEN = 32
cdef int BATCH_SIZE = 64
cdef double initialization_stddev = 0.01  # 权重初始化的方差
cdef int n_valid = 200
cdef int aux_dim = 4
cdef int num_env = 1
cdef double inf = 2147483647 / 2
#########################  embedding method ##########################################################
cdef int max_bp_iter = 3
cdef int aggregatorID = 0  #0:sum; 1:mean; 2:GCN
cdef int embeddingMethod = 1  #0:structure2vec; 1:graphsage


class GraphSAGELayer(nn.Module):
    def __init__(self, embedding_size):
        """
        Notes
        --------
        * Input : [node_cnt, embedding_size] - embeddings for all nodes in full graph
        * Input : [node_cnt, node_cnt] - full graph stored in Adjacent Matrix
        * Input : [batch_size, node_cnt] - each graph in batch has which nodes
        * Output : [node_cnt, embedding_size] - embeddings for all nodes in full graph
        * Output : [batch_size, embedding_size] - embeddings for virtual nodes
        """
        super().__init__()

        self.W_2 = nn.Parameter(truncated_normal([embedding_size, embedding_size]))
        self.W_3 = nn.Parameter(truncated_normal([embedding_size, embedding_size]))
        # to ensure size is int
        self.double_embedding_to_one = nn.Linear(embedding_size * 2, embedding_size, dtype=torch.float32)

    def forward(self,
                input_feature: Tensor,
                input_feature_virtual: Tensor,
                n2nsum_param: sparse.Tensor,
                subgsum_param: sparse.Tensor
                ):
        # [] = [node_cnt, node_cnt] * [node_cnt, embedding_size]
        h_neighbour = sparse.mm(n2nsum_param, input_feature).to(torch.float32)
        # [] = [batch_size, node_cnt] * [node_cnt, embedding_size]
        h_neighbour_virtual = sparse.mm(subgsum_param, input_feature).to(torch.float32)

        # [node_cnt, embedding_size*2]
        h_v = torch.concat(
            [
                torch.matmul(input_feature, self.W_2),
                torch.matmul(h_neighbour, self.W_3)
            ],
            dim=1
        ).to(torch.float32)

        # [batch_size, embedding_size*2]
        h_v_virtual = torch.concat(
            [
                torch.matmul(input_feature_virtual, self.W_2),
                torch.matmul(h_neighbour_virtual, self.W_3)
            ],
            dim=1
        ).to(torch.float32)

        h_v = relu(h_v)
        h_v_virtual = relu(h_v_virtual)
        h_v = self.double_embedding_to_one(h_v)
        h_v_virtual = self.double_embedding_to_one(h_v_virtual)
        h_v = normalize(h_v)
        h_v_virtual = normalize(h_v_virtual)
        return h_v.to(torch.float32), h_v_virtual.to(torch.float32)


class FINDER(nn.Module):
    def __init__(self, k=3, embedding_size=EMBEDDING_SIZE, batch_size=BATCH_SIZE, reg_hidden=REG_HIDDEN,
                 aux_dim=aux_dim, is_target_network=False, aggregator_id=aggregatorID, device=torch.device("cuda")):
        super().__init__()

        # init some parameters
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.reg_hidden = reg_hidden
        self.k = k
        self.is_target_network = is_target_network
        self.aggregator_id = aggregator_id
        self.aux_dim = aux_dim

        self.W_1 = nn.Parameter(truncated_normal([2, embedding_size]))
        # W_2 and W_3 in GraphSAGE
        self.W_4 = nn.Parameter(truncated_normal([embedding_size]))
        if reg_hidden > 0:
            self.hidden = nn.Parameter(truncated_normal([embedding_size, reg_hidden]))
            self.W_5 = nn.Parameter(truncated_normal([reg_hidden + aux_dim, 1]))
        else:
            self.W_5 = nn.Parameter(truncated_normal([embedding_size + aux_dim, 1]))
        self.GraphSAGE = GraphSAGELayer(embedding_size)

        # prepare batch
        self.inputs = {
            "action_select": None,
            "rep_global": None,
            "n2nsum_param": None,
            "subgsum_param": None,
            "target": None,
            "aux_input": None
        }
        self.device = device

    def forward(self,
                input_feature: Tensor = None,
                input_feature_virtual: Tensor = None,
                action_select: sparse.Tensor = None,  # [batch_size, node_cnt]
                rep_global: sparse.Tensor = None,  # [node_cnt, batch_size]
                n2nsum_param: sparse.Tensor = None,  # [node_cnt, node_cnt]
                subgsum_param: sparse.Tensor = None,  # [batch_size, node_cnt]
                aux_input: Tensor = None,  # [batch_size, aux_dim]
                laplacian_param: sparse.Tensor = None
                ):
        on, ov = self.ones_as_input_feature(n2nsum_param, subgsum_param, self.device)
        if input_feature_virtual == None:
            input_feature_virtual = ov.detach().clone()
        if input_feature == None:
            input_feature = on.detach().clone()

        n2nsum_param = n2nsum_param.to(self.device)
        subgsum_param = subgsum_param.to(self.device)
        aux_input = torch.tensor(aux_input, device=self.device, dtype=torch.float32)

        # [] = [node_cnt, input_size] * [input_size, embedding_size]
        h_0 = relu(torch.matmul(input_feature, self.W_1))
        h_0 = normalize(h_0)
        # [batch_size, embedding_size] initial methods should be same as input_feature
        h_0_virtual = relu(torch.matmul(input_feature_virtual, self.W_1))
        h_0_virtual = normalize(h_0_virtual)

        h_v = h_0
        h_v_virtual = h_0_virtual
        for i in range(self.k):
            # [node_cnt, embedding_size], [batch_size, embedding_size]
            h_v, h_v_virtual = self.GraphSAGE(h_v, h_v_virtual, n2nsum_param, subgsum_param)

        if action_select is not None:
            action_select = action_select.to(self.device)
            rep_global = rep_global.to(self.device)

            laplacian_param = laplacian_param.to(self.device)

            # [batch_size, embedding_size] = [batch_size, node_cnt] * [node_cnt, embedding_size]
            after_action_embedding = sparse.mm(action_select, h_v)

            # [batch_size, embed_dim, embed_dim] = [batch_size, embedding_size, 1] * [batch_size, 1, embedding_size]
            # here perform part of Q function: for every z_a and z_s, get outer product, z_a^T * z_s
            # the result `temp` has batch_size of [embedding_size, embedding_size] matrices
            temp = torch.matmul(torch.unsqueeze(after_action_embedding, dim=2), torch.unsqueeze(h_v_virtual, dim=1))

            # input of Relu in Q function
            # [batch_size, embedding_size, 1]
            embed_s_a = torch.matmul(
                temp,  # [batch_size, embed_dim, embed_dim] z_a^T * z_s
                torch.unsqueeze(  # [batch_size, embedding_size, 1] later multiply W_4 to each z_a^T * z_s
                    torch.tile(self.W_4, [self.batch_size, 1]),
                    dim=2
                )
            )
            # reshape [batch_size, embedding_size, 1] to [batch_size, embedding_size]
            embed_s_a = torch.reshape(
                embed_s_a,
                [self.batch_size, self.embedding_size]
            )

            if self.reg_hidden > 0:
                hidden_out = torch.matmul(embed_s_a, self.hidden)
                relu_out = relu(hidden_out)
            else:
                relu_out = relu(embed_s_a)

            q_value = torch.matmul(
                torch.concat([relu_out, aux_input], dim=1),
                self.W_5
            )

            loss_recons = 2 * torch.trace(
                torch.matmul(
                    h_v.T,
                    sparse.mm(laplacian_param, h_v)
                )
            )
            edge_num = sparse.sum(n2nsum_param)
            loss_recons = loss_recons / edge_num

            return q_value.to(self.device), loss_recons.to(self.device)
        elif rep_global is not None:
            rep_global = rep_global.to(self.device)

            rep_virtual = sparse.mm(rep_global, h_v_virtual)

            temp = torch.matmul(
                torch.unsqueeze(h_v, dim=2),
                torch.unsqueeze(rep_virtual, dim=1)
            )

            embed_s_a_all = torch.reshape(
                torch.matmul(
                    temp,  # [node_cnt, embed_dim, embed_dim]
                    torch.unsqueeze(  # [node_cnt, embedding_size, 1]
                        torch.tile(self.W_4, [h_v.shape[0], 1]),
                        dim=2
                    )
                ),
                h_v.shape
            )

            if self.reg_hidden > 0:
                #[node_cnt, 2 * embed_dim] * [2 * embed_dim, reg_hidden] = [node_cnt, reg_hidden1]
                hidden_out = torch.matmul(embed_s_a_all, self.hidden)
                #Relu, [node_cnt, reg_hidden1]
                last_output = relu(hidden_out)
            else:
                last_output = relu(embed_s_a_all)

            rep_aux = sparse.mm(rep_global, aux_input)
            last_output = torch.concat([last_output, rep_aux], 1)
            q_on_all = torch.matmul(last_output, self.W_5)
            return q_on_all.to(self.device)
        else:
            raise ValueError("Wrong input of forward")

    def predict(self, g_list, covered):
        cdef int n_graphs = len(g_list)
        cdef int i, j, k, bsize
        for i in range(0, n_graphs, BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j - i] = j
            batch_idxes = np.int32(batch_idxes)

            idx_map_list = self.setup_pred_all(batch_idxes, g_list, covered)
            # my_dict[self.rep_global] = self.inputs['rep_global']
            # my_dict[self.n2nsum_param] = self.inputs['n2nsum_param']
            # my_dict[self.subgsum_param] = self.inputs['subgsum_param']
            # my_dict[self.aux_input] = np.array(self.inputs['aux_input'])

            assert (self.inputs['rep_global'] is not None)
            assert (self.inputs['n2nsum_param'] is not None)
            assert (self.inputs['subgsum_param'] is not None)
            assert (self.inputs['aux_input'] is not None)

            self.eval()
            with torch.no_grad():
                result = self.forward(
                    rep_global=self.inputs['rep_global'],
                    n2nsum_param=self.inputs['n2nsum_param'],
                    subgsum_param=self.inputs['subgsum_param'],
                    aux_input=self.inputs['aux_input']
                )

            pos = 0
            pred = []
            for j in range(i, i + bsize):
                idx_map = idx_map_list[j - i]
                cur_pred = np.zeros(len(idx_map))
                for k in range(len(idx_map)):
                    if idx_map[k] < 0:
                        cur_pred[k] = -inf
                    else:
                        cur_pred[k] = result[pos]
                        pos += 1
                for k in covered[j]:
                    cur_pred[k] = -inf
                pred.append(cur_pred)
            assert (pos == len(result))
        return pred

    def fit(self, g_list, covered, actions, list_target):
        cdef int n_graphs = len(g_list)
        cdef int i, j, bsize

        for i in range(0, n_graphs, BATCH_SIZE):
            bsize = BATCH_SIZE
            if (i + BATCH_SIZE) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            # batch_idxes = []
            for j in range(i, i + bsize):
                batch_idxes[j - i] = j
                # batch_idxes.append(j)
            batch_idxes = np.int32(batch_idxes)
            """
            Check c++ module function: <PrepareBatchGraph::SetupGraphInput> for detailed info
            """
            # print("size of target in FINDER.fit(): ", list_target.shape)

            self.setup_train(batch_idxes, g_list, covered, actions, list_target)

            # print("size of target in FINDER.fit() after setup: ", self.inputs['target'])
            # my_dict[self.action_select] = self.inputs['action_select']
            # my_dict[self.rep_global] = self.inputs['rep_global']
            # my_dict[self.n2nsum_param] = self.inputs['n2nsum_param']
            # my_dict[self.laplacian_param] = self.inputs['laplacian_param']
            # my_dict[self.subgsum_param] = self.inputs['subgsum_param']
            # my_dict[self.aux_input] = np.array(self.inputs['aux_input'])
            # my_dict[self.target] = self.inputs['target']
            result, loss_recons = self.forward(
                action_select=self.inputs['action_select'],
                rep_global=self.inputs['rep_global'],
                n2nsum_param=self.inputs['n2nsum_param'],
                subgsum_param=self.inputs['subgsum_param'],
                aux_input=self.inputs['aux_input'],
                laplacian_param=self.inputs['laplacian_param']
            )
            # print("size of target in FINDER.fit() after forward: ", self.inputs['target'])
            loss_rl = mse_loss(result, torch.tensor(self.inputs['target'], device=self.device, dtype=torch.float32))
            loss = loss_rl + Alpha * loss_recons

            return loss.to(torch.float32)

    def setup_pred_all(self, idxes, g_list, covered):
        prepare_batch_graph = PrepareBatchGraph.py_PrepareBatchGraph(self.aggregator_id)
        prepare_batch_graph.SetupPredAll(idxes, g_list, covered)
        self.inputs['rep_global'] = prepare_batch_graph.rep_global
        self.inputs['n2nsum_param'] = prepare_batch_graph.n2nsum_param
        # self.inputs['laplacian_param'] = prepareBatchGraph.laplacian_param
        self.inputs['subgsum_param'] = prepare_batch_graph.subgsum_param
        self.inputs['aux_input'] = prepare_batch_graph.aux_feat
        return prepare_batch_graph.idx_map_list

    def setup_train(self, idxes, g_list, covered, actions, target: Tensor):
        self.inputs['target'] = target
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(aggregatorID)
        prepareBatchGraph.SetupTrain(idxes, g_list, covered, actions)
        self.inputs['action_select'] = prepareBatchGraph.act_select
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2nsum_param'] = prepareBatchGraph.n2nsum_param
        self.inputs['laplacian_param'] = prepareBatchGraph.laplacian_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        self.inputs['aux_input'] = prepareBatchGraph.aux_feat

    @staticmethod
    def ones_as_input_feature(n2nsum_param: Tensor, subgsum_param: Tensor, device):
        #[node_cnt, 2]
        nodes_size = n2nsum_param.shape[0]
        nodes_input = torch.ones((nodes_size, 2), device=device, dtype=torch.float32)
        # [batch_size, 2]
        y_nodes_size = subgsum_param.shape[0]
        y_nodes_input = torch.ones((subgsum_param.shape[0], 2), device=device, dtype=torch.float32)
        return nodes_input, y_nodes_input

