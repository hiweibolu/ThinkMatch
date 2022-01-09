# 参考给出的PCA的gconv_pdl.py修改，与pytorch版的gconv.py合并

import math
import time
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Gconv(nn.Layer):
    """
    (Intra) graph convolution operation, with single convolutional layer
    """
    def __init__(self, in_features, out_features):
        super(Gconv, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        k = math.sqrt(1.0 / in_features)
        weight_attr_1 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        bias_attr_1 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        weight_attr_2 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        bias_attr_2 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))

        self.a_fc = nn.Linear(self.num_inputs, self.num_outputs,
                                weight_attr=weight_attr_1, 
                                bias_attr=bias_attr_1)
        self.u_fc = nn.Linear(self.num_inputs, self.num_outputs,
                                weight_attr=weight_attr_2, 
                                bias_attr=bias_attr_2)

    def forward(self, A, x, norm=True):
        if norm is True:
            A = F.normalize(A, p=1, axis=-2)
    
        '''
        st = time.time()
        ax = self.a_fc(x)
        ux = self.u_fc(x)
        print('Twp Linear layer cost {}s'.format(time.time() - st))
        
        st = time.time()
        x = paddle.bmm(A, F.relu(ax))  # has size (bs, N, num_outputs)
        print('bmm + relu cost {}s'.format(time.time() - st))

        st = time.time()
        x += F.relu(ux)
        print('+= relu() cost {}s'.format(time.time() - st))

        '''
        ax = self.a_fc(x)
        ux = self.u_fc(x)
        x = paddle.bmm(A, F.relu(ax)) + F.relu(ux) # has size (bs, N, num_outputs)
        return x

class Siamese_ChannelIndependentConv(nn.Layer):
    r"""
    Siamese Channel Independent Conv neural network for processing arbitrary number of graphs.

    :param in_features: the dimension of input node features
    :param num_features: the dimension of output node features
    :param in_edges: the dimension of input edge features
    :param out_edges: (optional) the dimension of output edge features. It needs to be the same as ``num_features``
    """
    def __init__(self, in_features, num_features, in_edges, out_edges=None):
        super(Siamese_ChannelIndependentConv, self).__init__()
        self.in_feature = in_features
        self.gconv = ChannelIndependentConv(in_features, num_features, in_edges, out_edges)

    #def forward(self, g1: Tuple[Tensor, Tensor, Optional[bool]], *args) -> List[Tensor]:
    def forward(self, g1, *args):
        r"""
        Forward computation of Siamese Channel Independent Conv.

        :param g1: The first graph, which is a tuple of (:math:`(b\times n\times n)` {0,1} adjacency matrix,
         :math:`(b\times n\times d_n)` input node embedding, :math:`(b\times n\times n\times d_e)` input edge embedding,
         mode (``1`` or ``2``))
        :param args: Other graphs
        :return: A list of tensors composed of new node embeddings :math:`(b\times n\times d^\prime)`, appended with new
         edge embeddings :math:`(b\times n\times n\times d^\prime)`
        """
        emb1, emb_edge1 = self.gconv(*g1)
        embs = [emb1]
        emb_edges = [emb_edge1]
        for g in args:
            emb2, emb_edge2 = self.gconv(*g)
            embs.append(emb2), emb_edges.append(emb_edge2)
        return embs + emb_edges
