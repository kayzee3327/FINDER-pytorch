import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import relu, normalize, huber_loss, mse_loss
from torch import sparse



# Hyper Parameters:
cdef double GAMMA = 1  # decay rate of past observations
cdef int UPDATE_TIME = 1000
cdef int EMBEDDING_SIZE = 64
cdef int MAX_ITERATION = 500000
cdef double LEARNING_RATE = 0.0001   #dai
cdef int MEMORY_SIZE = 500000
cdef double Alpha = 0.001 ## weight of reconstruction loss
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
cdef double inf = 2147483647/2
#########################  embedding method ##########################################################
cdef int max_bp_iter = 3
cdef int aggregatorID = 0 #0:sum; 1:mean; 2:GCN
cdef int embeddingMethod = 1   #0:structure2vec; 1:graphsage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FINDERconv(nn.Module):
    def __init__(self):
        super().__init__()

        # init some parameters
        self.embedding_size = EMBEDDING_SIZE
        self.reg_hidden = REG_HIDDEN

    def __truncated_normal(self, size, requires_grad=True, dtype=torch.float32) -> torch.Tensor:
        """

        :param size: tensor size
        :param requires_grad: tensor "requires_grad" parameter
        :param dtype: tensor "dtype" parameter
        :return: An initialized tensor with truncated normal data.
            || Range: [-2*initialization_stddev, 2*initialization_stddev]
            || Mean: 0
            || Standard Deviation: initialization_stddev
        """

        t = torch.empty(size, requires_grad=requires_grad, dtype=dtype)

        return init.trunc_normal_(
            t,
            std=initialization_stddev,
            a=(-2.0)*initialization_stddev,
            b=2.0*initialization_stddev
        ).to(device)
    def forward(self,
                action_select: torch.Tensor,      # [batch_size, node_cnt]
                rep_global: torch.Tensor,         # [node_cnt, batch_size]
                n2nsum_param: torch.Tensor,       # [node_cnt, node_cnt]
                subgsum_param: torch.Tensor,      # [batch_size, node_cnt]
                target: torch.Tensor,             # [batch_size,1]
                aux_input: torch.Tensor           # [batch_size, aux_dim]

    ):
        # [2, embed_dim], initialization as tf.truncated_normal()
        w_n2l = self.__truncated_normal([2, self.embedding_size])
        # [embed_dim, embed_dim]
        p_node_conv = self.__truncated_normal([self.embedding_size, self.embedding_size])

        if embeddingMethod == 1:    #'graphsage'
            # [embed_dim, embed_dim]
            p_node_conv2 = self.__truncated_normal([self.embedding_size, self.embedding_size])
            # [2*embed_dim, embed_dim]
            p_node_conv3 = self.__truncated_normal([2*self.embedding_size, self.embedding_size])

        #[reg_hidden+aux_dim, 1]
        if self.reg_hidden > 0:
            #[2*embed_dim, reg_hidden]
            # h1_weight = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            # [embed_dim, reg_hidden]
            h1_weight = self.__truncated_normal([self.embedding_size, self.reg_hidden])
            #[reg_hidden1, reg_hidden2]
            # h2_weight = tf.Variable(tf.truncated_normal([self.reg_hidden1, self.reg_hidden2], stddev=initialization_stddev), tf.float32)
            #[reg_hidden+aux_dim, 1]
            h2_weight = self.__truncated_normal([self.reg_hidden + aux_dim, 1])
            #[reg_hidden2 + aux_dim, 1]
            last_w = h2_weight
        else:
            #[2*embed_dim, reg_hidden]
            h1_weight = self.__truncated_normal([2*self.embedding_size, self.reg_hidden])
            # [embed_dim, reg_hidden]
            # h1_weight = tf.Variable(tf.truncated_normal([self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            #[2*embed_dim, reg_hidden]
            last_w = h1_weight

        ## [embed_dim, 1]
        cross_product = self.__truncated_normal([self.embedding_size, 1])

        nodes_size = n2nsum_param.size()[0]
        #[node_cnt, 2]
        node_input = torch.ones((nodes_size, 2)).to(device)
        '''
        ones() is like even attention on neighborhood?
        '''
        y_nodes_size = subgsum_param.size()[0]
        # [batch_size, 2]
        y_node_input = torch.ones((y_nodes_size,2)).to(device)

        # [node_cnt, 2] * [2, embed_dim] = [node_cnt, embed_dim]
        input_message = torch.matmul(node_input.type(torch.float32), w_n2l).to(device)

        # [node_cnt, embed_dim]  # no sparse
        input_potential_layer = relu(input_message).to(device)

        # no sparse
        # [batch_size, embed_dim]
        y_input_message = torch.matmul(y_node_input.type(torch.float32), w_n2l).to(device)
        #[batch_size, embed_dim]  # no sparse
        y_input_potential_layer = relu(y_input_message).to(device)

        #input_potential_layer = input_message
        cdef int lv = 0
        #[node_cnt, embed_dim], no sparse
        cur_message_layer = input_potential_layer
        cur_message_layer = normalize(cur_message_layer).to(device)

        #[batch_size, embed_dim], no sparse
        y_cur_message_layer = y_input_potential_layer
        y_cur_message_layer = normalize(y_cur_message_layer).to(device)

        while lv < max_bp_iter:
            lv = lv + 1

            #[node_cnt, node_cnt] * [node_cnt, embed_dim] = [node_cnt, embed_dim], dense
            n2npool = sparse.mm(n2nsum_param.type(torch.float32), cur_message_layer).to(device)
            #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
            node_linear = torch.matmul(n2npool, p_node_conv).to(device)

            # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
            y_n2npool = sparse.mm(n2nsum_param.type(torch.float32), cur_message_layer).to(device)
            #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
            y_node_linear = torch.matmul(y_n2npool, p_node_conv).to(device)

            if embeddingMethod == 0: # 'structure2vec'
                #[node_cnt, embed_dim] + [node_cnt, embed_dim] = [node_cnt, embed_dim], return tensed matrix
                merged_linear = torch.add(node_linear, input_message).to(device)
                #[node_cnt, embed_dim]
                cur_message_layer = relu(merged_linear).to(device)

                #[batch_size, embed_dim] + [batch_size, embed_dim] = [batch_size, embed_dim], return tensed matrix
                y_merged_linear = torch.add(y_node_linear, y_input_message).to(device)
                #[batch_size, embed_dim]
                y_cur_message_layer = relu(y_merged_linear).to(device)
            else:   # 'graphsage'
                #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
                cur_message_layer_linear = torch.matmul(cur_message_layer.type(torch.float32), p_node_conv2).to(device)

                #[[node_cnt, embed_dim] [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim], return tensed matrix
                merged_linear = torch.concat([node_linear, cur_message_layer_linear], 1).to(device)
                #[node_cnt, 2*embed_dim]*[2*embed_dim, embed_dim] = [node_cnt, embed_dim]
                cur_message_layer = relu(torch.matmul(merged_linear, p_node_conv3)).to(device)

                #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
                y_cur_message_layer_linear = torch.matmul(y_cur_message_layer.type(torch.float32), p_node_conv2).to(device)
                #[[batch_size, embed_dim] [batch_size, embed_dim]] = [batch_size, 2*embed_dim], return tensed matrix
                y_merged_linear = torch.concat([y_node_linear, y_cur_message_layer_linear], 1).to(device)
                #[batch_size, 2*embed_dim]*[2*embed_dim, embed_dim] = [batch_size, embed_dim]
                y_cur_message_layer = relu(torch.matmul(y_merged_linear, p_node_conv3)).to(device)

            cur_message_layer = normalize(cur_message_layer).to(device)
            y_cur_message_layer = normalize(y_cur_message_layer).to(device)

        # self.node_embedding = cur_message_layer
        # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim], dense
        # y_potential = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param,tf.float32), cur_message_layer)
        y_potential = y_cur_message_layer
        #[batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
        action_embed = sparse.mm(action_select.type(torch.float32), cur_message_layer).to(device)

        # embed_s_a = tf.concat([action_embed,y_potential],1)

        '''
        Note that unlike tf.tile(), torch.tile() does not require a torch.unsqueeze() operation beforehand to reserve enough dimensions.
        In accordance to original FINDER project, torch.unsqueeze() is added.
        '''
        # [batch_size, embed_dim, embed_dim]
        temp = torch.matmul(torch.unsqueeze(action_embed, 2), torch.unsqueeze(y_potential, 1)).to(device)
        Shape = action_embed.size()
        embed_s_a = torch.reshape(torch.matmul(temp, torch.reshape(torch.tile(cross_product, [Shape[0],1]), [Shape[0],Shape[1],1])), Shape).to(device)

        #[batch_size, embed_dim]
        last_output = embed_s_a

        if self.reg_hidden > 0:
            #[batch_size, 2*embed_dim] * [2*embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
            hidden = torch.matmul(embed_s_a, h1_weight).to(device)
            #[batch_size, reg_hidden]
            last_output = relu(hidden).to(device)

        # if reg_hidden == 0: ,[[batch_size, 2*embed_dim], [batch_size, aux_dim]] = [batch_size, 2*embed_dim+aux_dim]
        # if reg_hidden > 0: ,[[batch_size, reg_hidden], [batch_size, aux_dim]] = [batch_size, reg_hidden+aux_dim]
        last_output = torch.concat([last_output, self.aux_input], 1).to(device)
        #if reg_hidden == 0: ,[batch_size, 2*embed_dim+aux_dim] * [2*embed_dim+aux_dim, 1] = [batch_size, 1]
        #if reg_hidden > 0: ,[batch_size, reg_hidden+aux_dim] * [reg_hidden+aux_dim, 1] = [batch_size, 1]
        q_pred = torch.matmul(last_output, last_w).to(device)

        #[node_cnt, batch_size] * [batch_size, embed_dim] = [node_cnt, embed_dim]
        rep_y = sparse.mm(rep_global.type(torch.float32), y_potential).to(device)

        # embed_s_a_all = tf.concat([cur_message_layer,rep_y],1)

        # [node_cnt, embed_dim, embed_dim]
        temp1 = torch.matmul(torch.unsqueeze(cur_message_layer, dim=2), torch.unsqueeze(rep_y, dim=1)).to(device)
        # [node_cnt embed_dim]
        Shape1 = cur_message_layer.size()
        # [batch_size, embed_dim], first transform
        embed_s_a_all = torch.reshape(
            torch.matmul(temp1, torch.reshape(torch.tile(cross_product, [Shape1[0], 1]), [Shape1[0], Shape1[1], 1])), Shape1).to(device)

        #[node_cnt, 2 * embed_dim]
        last_output = embed_s_a_all
        if self.reg_hidden > 0:
            #[node_cnt, 2 * embed_dim] * [2 * embed_dim, reg_hidden] = [node_cnt, reg_hidden1]
            hidden = torch.matmul(embed_s_a_all, h1_weight).to(device)
            #Relu, [node_cnt, reg_hidden1]
            last_output = relu(hidden).to(device)
            #[node_cnt, reg_hidden1] * [reg_hidden1, reg_hidden2] = [node_cnt, reg_hidden2]


        #[node_cnt, batch_size] * [batch_size, aux_dim] = [node_cnt, aux_dim]
        rep_aux = sparse.mm(rep_global.type(torch.float32), self.aux_input).to(device)

        #if reg_hidden == 0: , [[node_cnt, 2 * embed_dim], [node_cnt, aux_dim]] = [node_cnt, 2*embed_dim + aux_dim]
        #if reg_hidden > 0: , [[node_cnt, reg_hidden], [node_cnt, aux_dim]] = [node_cnt, reg_hidden + aux_dim]
        last_output = torch.concat([last_output, rep_aux], 1).to(device)

        #if reg_hidden == 0: , [node_cnt, 2 * embed_dim + aux_dim] * [2 * embed_dim + aux_dim, 1] = [node_cnt，1]
        #f reg_hidden > 0: , [node_cnt, reg_hidden + aux_dim] * [reg_hidden + aux_dim, 1] = [node_cnt，1]
        q_on_all = torch.matmul(last_output, last_w).to(device)
        TD_errors = torch.sum(torch.abs(target-q_pred), dim=1)
        return q_pred, q_on_all, cur_message_layer, TD_errors


'''Write loss function as class to keep necessary information'''
class LossFn(nn.Module):
    def __init__(self, IsPrioritizedSampling, IsHuberloss):
        super().__init__()
        self.IsPrioritizedSampling = IsPrioritizedSampling
        self.IsHuberloss = IsHuberloss

    def __squared_difference(self, pred, y):
        return torch.square(pred-y).to(device)
    def forward(self,

                laplacian_param,
                n2nsum_param,
                target,
                ISWeights,

                cur_message_layer,
                q_pred
        ):
        loss_recons = 2 * torch.trace(torch.matmul(torch.transpose(cur_message_layer), sparse.mm(laplacian_param.type(torch.float32), cur_message_layer))).to(device)
        edge_num = sparse.sum(n2nsum_param.type(torch.float32)).to(device)
        loss_recons = torch.divide(loss_recons, edge_num).to(device)

        if self.IsPrioritizedSampling:

            if self.IsHuberloss:
                loss_rl = huber_loss(ISWeights * target, ISWeights * q_pred)
            else:
                loss_rl = torch.mean(ISWeights * self.__squared_difference(q_pred, target))
        else:
            if self.IsHuberloss:
                loss_rl = huber_loss(q_pred, target)
            else:
                loss_rl = mse_loss(q_pred, target)

        loss = loss_rl + Alpha * loss_recons
        return loss