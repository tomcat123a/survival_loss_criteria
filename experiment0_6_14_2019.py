# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:23:36 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:57:33 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:29:04 2019

@author: Administrator
"""
import os
import torch
import torch.nn as nn
import time


import pandas as pd

import torch.nn.functional as F
 
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader,DataLoader
from torch_geometric.nn import GCNConv,ChebConv
from torch_geometric.nn import GraphConv,GatedGraphConv,GATConv,AGNNConv,ARMAConv,SGConv,APPNP,RGCNConv
from torch_geometric.nn import SignedConv,GMMConv
from torch_geometric.nn import SplineConv
from torch_geometric.nn import global_sort_pool,GlobalAttention,Set2Set
from torch_geometric.nn import TopKPooling,SAGEConv
from torch.nn import  LeakyReLU,ReLU,ELU
from torch import randperm
from torch_geometric.nn import max_pool,avg_pool,max_pool_x,avg_pool_x,graclus
from torch_geometric.nn import JumpingKnowledge,DeepGraphInfomax
from torch_geometric.nn import InnerProductDecoder,GAE,VGAE,ARGA,ARGVA

from torch_geometric.data import Data, DataLoader
import networkx as nx
import numpy as np

from torch_geometric.utils import from_networkx

from torch_geometric.utils import is_undirected, to_undirected

from torch_geometric.nn import global_mean_pool,global_max_pool
#####################################custom dense
import torch
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.nn.inits import uniform
 

class DenseSAGEConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.SAGEConv`.

    :rtype: :class:`Tensor`
    """

    def __init__(self, in_channels, out_channels, normalize=True, bias=True):
        super(DenseSAGEConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)


    def forward(self, x, adj, mask=None, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`.
            mask (ByteTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = x.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1

        #out = torch.matmul(adj, x)
        out = torch.bmm( adj.expand(B,-1,-1),x )
        #out = torch.cat([torch.matmul(adj,x[i]) for i in range(B)],dim=0)
        out = out / adj.sum(dim=-1, keepdim=True).clamp(min=1)
        out = torch.matmul(out, self.weight)

        if self.bias is not None:
            out = out + self.bias

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        if mask is not None:
            mask = mask.view(B, N, 1).to(x.dtype)
            out = out * mask

        return out


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



dense_diff_pool_EPS = 1e-15


def dense_diff_pool(x, adj, s, mask=None,link_loss_on=False,mini_batch=False):
    r"""Differentiable pooling operator from the `"Hierarchical Graph
    Representation Learning with Differentiable Pooling"
    <https://arxiv.org/abs/1806.08804>`_ paper
    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}
        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})
    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns pooled node feature matrix, coarsened adjacency matrix and the
    auxiliary link prediction objective :math:`\| \mathbf{A} -
    \mathrm{softmax}(\mathbf{S}) \cdot {\mathrm{softmax}(\mathbf{S})}^{\top}
    \|_F`.
    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
            \times N \times F}` with batch-size :math:`B`, (maximum)
            number of nodes :math:`N` for each graph, and feature dimension
            :math:`F`.
        adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
            \times N \times N}`.
        s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
            \times N \times C}` with number of clusters :math:`C`. The softmax
            does not have to be applied beforehand, since it is executed
            within this method.
        mask (ByteTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`)
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    #torch.matmul(s.transpose(1, 2), adj)
    m1=torch.bmm(s.transpose(1, 2), adj.expand(batch_size,-1,-1))
    out_adj = torch.matmul(m1, s)

    #link_loss = adj - torch.matmul(s, s.transpose(1, 2)).sum(dim=0)
    #link_loss = 
    #t0=time.time()
    if link_loss_on==True:
        if mini_batch==True:
            link_loss = torch.sqrt(torch.stack([torch.norm(adj - torch.matmul(s[i], s[i].transpose(0, 1)), p=2)**2 for i in torch.randint(batch_size,(min(10,batch_size),))]).sum(dim=0))
        else:
            link_loss = torch.sqrt(torch.stack([torch.norm(adj - torch.matmul(s[i], s[i].transpose(0, 1)), p=2)**2 for i in range(batch_size)]).sum(dim=0))
        
    #print(time.time()-t0)
    #t0=time.time()
    #link_loss = torch.sqrt(torch.stack([torch.norm(adj - torch.matmul(i, i.transpose(0, 1)), p=2)**2 for i in torch.unbind(s,dim=0)]).sum(dim=0))
    
    #print(time.time()-t0)
        link_loss = link_loss / adj.numel()
    else:
        link_loss = torch.tensor(0)

    ent_loss = (-s * torch.log(s + dense_diff_pool_EPS)).sum(dim=-1).mean()

    return out, out_adj, link_loss, ent_loss





###############losss


#GAT only edge list


def R_set(x):
	'''Create an indicator matrix of risk sets, where T_j >= T_i.
	Note that the input data have been sorted in descending order.
	Input:
		x: a PyTorch tensor that the number of rows is equal to the number of samples.
	Output:
		indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
	'''
	n_sample = x.size(0)
	matrix_ones = torch.ones(n_sample, n_sample)
	indicator_matrix = torch.tril(matrix_ones)

	return(indicator_matrix)


def neg_par_log_likelihood(pred, ytime,yevent):#event=0,censored
    #ytime should be sorted with increasing order
    #yevent=torch.tensor(yevent)#
    #yevent=yevent.view(-1,1).type(torch.float)  
    #pred=pred.view(-1)
    n_observed = int(yevent.sum(0))  
    ytime_indicator = R_set(ytime)
    #ytime_indicator=y_m
	###if gpu is being used
    if torch.cuda.is_available():
	    ytime_indicator = ytime_indicator.cuda()
	###
    risk_set_sum = ytime_indicator.mm(torch.exp(pred)) 
    diff = pred - torch.log(risk_set_sum)
    yevent=yevent.view(-1,1).type(torch.float)  
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
    cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))
    if cost==nan:
        print(pred)
        print(exp(pred))
    return(cost)



def c_index(pred, ytime, yevent):
	#yevent=torch.tensor(yevent)
    #yevent = yevent.view(-1,1).type(torch.float)   
    #pred=pred.view(-1)
    n_sample = yevent.size(0)  
   # n_uncensored = yevent.sum(0)
    #ytime_indicator = R_set(ytime)
    #ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
	###T_i is uncensored
    uncensor_idx = (yevent == 1).nonzero()
    
    #zeros = torch.zeros(n_sample)
    #ytime_matrix[censor_idx, :] = zeros
	###1 if pred_i < pred_j; 0.5 if pred_i = pred_j
    #pred_matrix = torch.zeros_like(ytime_matrix)
    pred=pred.view(-1)
    denomi=(ytime.expand(n_sample,n_sample).transpose(0,1).triu()<ytime.expand(n_sample,n_sample).triu())[uncensor_idx].sum()
    nume=(pred.expand(n_sample,n_sample).transpose(0,1).triu()<pred.expand(n_sample,n_sample).triu())[uncensor_idx].sum()
    concordance_index = nume.cpu().numpy()/denomi.cpu().numpy()
	###if gpu is being used
    if torch.cuda.is_available():
        concordance_index = concordance_index.cuda()
	###
    return(concordance_index)

def get_data_list_el(x_matrix,el):  
    data_list=[]
    for i in range(1,x_matrix.shape[1]):
        
        
        data_list+=[Data(x=torch.tensor(x_matrix[:,i].reshape((-1,1))).type(torch.float32),edge_index=el)]
    return data_list



def get_data_list_el_w(x_matrix,el,eattr):  
    data_list=[]
    for i in range(1,x_matrix.shape[1]):
        data_list+=[Data(x=torch.tensor(x_matrix[:,i].reshape((-1,1))).type(torch.float32),\
                         edge_index=el,edge_attr=torch.from_numpy(eattr.reshape(-1)))]
    return data_list




def get_data_adj(x_matrix):  
     
    x_matrix=x_matrix[:,1:] 
    x=torch.from_numpy(x_matrix.astype(float)).type(torch.float32).t().contiguous().view(x_matrix.shape[1],x_matrix.shape[0],1)
    return x



def remove_bar(x):
    return x.split('|')[1]


 
def get_data_edge_lung(survpath,data_num,data_type='edge',train_rate=0.7,device='cpu'):
    
    if data_num<=3:
        #each row is a gene, and each column represents a sample
        x=pd.read_csv('{}/survdata/lung_x_{}.csv'.format(survpath,data_num))#
        y=pd.read_csv('{}/survdata/lung_t_{}.csv'.format(survpath,data_num)).iloc[:,1].values
        censor=pd.read_csv('{}/survdata/lung_c_{}.csv'.format(survpath,data_num)).iloc[:,1].values
        edl=pd.read_csv('{}/survdata/dt_{}.csv'.format(survpath,data_num)).iloc[:,1:]
        #G=nx.Graph()
        print('num_genes{}'.format(x.shape[0]))
        n=x.shape[1]-1
        #each column, except for the column 0 of x ,represents features for a sample
        #the features are in alphabetical order,this should agree with the order
        #in edl
        x=pd.concat([x.iloc[:,0],x.iloc[:,1:].take(np.argsort(y),axis=1)],axis=1)
        x=x.take(np.argsort(x.iloc[:,0].astype(str)),axis=0)
        y=y[np.argsort(y)]
        censor=censor[np.argsort(y)]#1 death uncensor,0 censor
    else:
        CANCERSET=['STAD','COAD','UCEC','THCA','KIRC','PRAD','LUAD','LUSC','SKCM','BLCA']
        x=pd.read_csv('{}/survdata/{}_x.csv'.format(survpath,CANCERSET[data_num-4]))
        y=pd.read_csv('{}/survdata/{}_t.csv'.format(survpath,CANCERSET[data_num-4])).iloc[:,1].values
        censor=pd.read_csv('{}/survdata/{}_c.csv'.format(survpath,CANCERSET[data_num-4])).iloc[:,1].values
        edl=pd.read_csv('{}/survdata/{}_network_0.csv'.format(survpath,CANCERSET[data_num-4])).iloc[:,1:]
        x = x.T.reset_index().iloc[1:,:]
        x.iloc[:,0]=list(map(remove_bar,x.iloc[:,0]))
        n=x.shape[1]-1
        x=pd.concat([x.iloc[:,0],x.iloc[:,1:].take(np.argsort(y),axis=1)],axis=1)
        x=x.take(np.argsort(x.iloc[:,0].astype(str)),axis=0)
        y=y[np.argsort(y)]
        censor=censor[np.argsort(y)] 
    if data_type=='edge':
        ev=edl.values
        
        edges_weight=[]
        for i in range(ev.shape[0]):
            edges_weight+=[(int(ev[i,0]),int(ev[i,1]),ev[i,2])]
        
       
        G=nx.Graph()
        G.add_nodes_from(x.iloc[:,0])
        #add self_loop with weight 1
        for i in set(G.nodes).difference(set(nx.isolates(G))):
            edges_weight+=[(i,i,1.)]
        
        G.add_weighted_edges_from(edges_weight)
        G.remove_nodes_from(list(nx.isolates(G)))
        G = nx.DiGraph(G) 
        
        adj_bi=nx.adjacency_matrix(G,weight=None).todense().astype(int)
        G1=nx.from_numpy_matrix(adj_bi)
        edges_tc = torch.from_numpy(np.array(G1.edges)).long().t().contiguous()
        data_list=get_data_list_el(x.values,edges_tc)
        #loader = DataLoader(data_list, batch_size=len(data_list),shuffle=False)
    elif data_type=='edge_weight':
    
        ev=edl.values
        
        edges_weight=[]
        for i in range(ev.shape[0]):
            edges_weight+=[(int(ev[i,0]),int(ev[i,1]),ev[i,2])]
        
        G=nx.Graph()
        G.add_nodes_from(x.iloc[:,0])
        #add self_loop with weight 1
        for i in G.nodes:
            edges_weight+=[(i,i,1.)]
        G.add_weighted_edges_from(edges_weight)
        #from un_directed to directed
        G = nx.DiGraph(G)
        adj_weight=nx.adjacency_matrix(G).todense()
        G1=nx.from_numpy_matrix(adj_weight)
        edges_weight=np.array([*G1.edges.data('weight')])[:,2]
        edges_tc = torch.from_numpy(np.array(G1.edges)).long().t().contiguous()
        #data_list=get_data_list_el(x.values,edges_tc)
        data_list=get_data_list_el_w(x.values,edges_tc,edges_weight)
        #loader = DataLoader(data_list, batch_size=len(data_list),shuffle=False)
    else:
        G=nx.Graph()
        #G.add_nodes_from([1,2,3])
        #G.add_weighted_edges_from([(1,2,0.5),(1,4,0.5)])
        
        G.add_nodes_from(x.iloc[:,0])
        ev=edl.values
       
        edges_weight=[]
        for i in range(ev.shape[0]):
            edges_weight+=[(int(ev[i,0]),int(ev[i,1]),ev[i,2])]
        #add self_loop with weight 1
        for i in G.nodes:
            edges_weight+=[(i,i,1.)]
        G.add_weighted_edges_from(edges_weight)
        G = nx.DiGraph(G)
        adj_bi=nx.adjacency_matrix(G,weight=None).todense()
        adj_train=[]
        adj_test=[]
        data_list=get_data_adj(x.values).type(torch.FloatTensor).to(device)
        for i in range(3):
            adj_weight=(nx.adjacency_matrix(G).todense())**(i+1)
            G1=nx.from_numpy_matrix(adj_weight)
            
            
            #adjt=torch.tensor(adj_weight.astype(np.float32)).type(torch.FloatTensor).to(device)
            adjt=torch.unsqueeze(torch.tensor(adj_weight.astype(np.float32)),0).type(torch.FloatTensor).to(device)
            adj_train.append(adjt**(i+1))
            adj_test.append(adjt**(i+1))
        #loader=DenseDataLoader(data_list, batch_size=len(data_list),shuffle=False)   
    trainset=np.sort(np.random.choice(n,size=int(train_rate*n),replace=False))
    testset=np.sort(np.setdiff1d(range(n),trainset))
    if data_type=='edge' or data_type=='edge_weight':
        adjt=torch.unsqueeze(torch.tensor(adj_bi).type(torch.FloatTensor),0)
        return data_list,y,censor,trainset,testset,len(G.nodes),adjt,adjt
    else:
        return data_list,y,censor,trainset,testset,x.shape[0],adj_train,adj_test
#create dataset
#with edge list
#with weight
 
#test
class GNN(torch.nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                  
                 normalize=False,
                 add_loop=False,
                 gnn_k=1,
                 gnn_type=1,
                 jump=None,#None,max,lstm
                 activation='leaky'
                 ):
        super(GNN, self).__init__()

        self.add_loop = add_loop

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.k=gnn_k#number of repitiions of gnn
        self.gnn_type=gnn_type
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.bn2 = nn.Sequential(*[torch.nn.BatchNorm1d(out_channels)]*(self.k-1))
        self.jump=jump
        if not( jump is None):
            if jump!='lstm':
                self.jk=JumpingKnowledge(jump)
            else:
                self.jk=JumpingKnowledge(jump,out_channels,self.k )
        if activation=='leaky':
            self.activ=F.leaky_relu
        elif activation=='elu':
            self.activ=F.elu
        elif activation=='relu':
            self.activ=F.relu
         
        
        if self.k ==1 and not( self.jump is  None):
            raise Exception('jumping knowledge only serves for the case where k>1!')
        if gnn_type==0:
            self.conv1 = DenseSAGEConv(in_channels=self.in_channels, out_channels=out_channels, normalize=False)
            self.conv2 = nn.Sequential(*[DenseSAGEConv(in_channels=out_channels, out_channels=out_channels, normalize=False)]*(self.k-1))
        
            #self.conv2 = nn.Sequential(*[DenseSAGEConv(in_channels=out_channels, out_channels=out_channels, normalize=False)]*(self.k-1))
        if gnn_type==1:
            self.conv1 = DenseSAGEConv(in_channels=self.in_channels, out_channels=out_channels, normalize=True)
            self.conv2 = nn.Sequential(*[DenseSAGEConv(in_channels=out_channels, out_channels=out_channels, normalize=True)]*(self.k-1))
        
        if gnn_type==2:
            self.conv1 = GCNConv(in_channels=self.in_channels, out_channels=out_channels, cached=False)
            self.conv2 = nn.Sequential(*[GCNConv(in_channels=out_channels, out_channels=out_channels, cached=False)]*(self.k-1))
        if gnn_type==3:
            self.conv1 = GCNConv(in_channels=self.in_channels, out_channels=out_channels,improved=True, cached=False)
            self.conv2 = nn.Sequential(*[GCNConv(in_channels=out_channels, out_channels=out_channels,improved=True, cached=False)]*(self.k-1))
        if gnn_type==4:
            self.conv1 = ChebConv(in_channels=self.in_channels, out_channels=out_channels,K=2)
            self.conv2 = nn.Sequential(*[ChebConv(in_channels=out_channels, out_channels=out_channels,K=2)]*(self.k-1))
        if gnn_type==5:
            self.conv1 = ChebConv(in_channels=self.in_channels, out_channels=out_channels,K=4)
            self.conv2 = nn.Sequential(*[ChebConv(in_channels=out_channels, out_channels=out_channels,K=4)]*(self.k-1))
        if gnn_type==6:
            self.conv1 = GraphConv(in_channels=self.in_channels, out_channels=out_channels,aggr='add')
            self.conv2 = nn.Sequential(*[GraphConv(in_channels=out_channels, out_channels=out_channels,aggr='add')]*(self.k-1))
        if gnn_type==7:
            self.conv1 = GatedGraphConv(out_channels=out_channels, num_layers=3, aggr='add', bias=True)
            self.conv2 = nn.Sequential(*[GatedGraphConv(out_channels=out_channels, num_layers=3, aggr='add', bias=True)]*(self.k-1))
        if gnn_type==8:
            self.conv1 = GatedGraphConv(out_channels=out_channels, num_layers=7, aggr='add', bias=True)
            self.conv2 = nn.Sequential(*[GatedGraphConv(out_channels=out_channels, num_layers=7, aggr='add', bias=True)]*(self.k-1))
        if gnn_type==9:
            self.conv1 =GATConv(in_channels=self.in_channels,out_channels=out_channels, heads=1, concat=True, negative_slope=0.2,dropout=0)
            self.conv2 =nn.Sequential(*[GATConv(in_channels=out_channels,out_channels=out_channels, heads=1, concat=True, negative_slope=0.2,dropout=0.6)]*(self.k-1))
        if gnn_type==10:
            self.conv1 =GATConv(in_channels=self.in_channels,out_channels=out_channels, heads=6, concat=False, negative_slope=0.2,dropout=0.6)
            self.conv2 =nn.Sequential(*[GATConv(in_channels=out_channels,out_channels=out_channels, heads=6, concat=False, negative_slope=0.2,dropout=0.6)]*(self.k-1))
            
        if gnn_type==11:
            self.conv1 =GATConv(in_channels=self.in_channels,out_channels=out_channels, heads=4, concat=True, negative_slope=0.2,dropout=0)
            self.conv2 =nn.Sequential(*[GATConv(in_channels=out_channels,out_channels=out_channels, heads=4, concat=True, negative_slope=0.2,dropout=0.6)]*(self.k-1))
        
        if gnn_type==12:
            self.conv1 =GATConv(in_channels=self.in_channels,out_channels=out_channels, heads=4, concat=False, negative_slope=0.2,dropout=0.6)
            self.conv2 =nn.Sequential(*[GATConv(in_channels=out_channels,out_channels=out_channels, heads=4, concat=False, negative_slope=0.2,dropout=0.6)]*(self.k-1))
            
        if gnn_type==13:
            self.conv1 = AGNNConv(requires_grad=True)
            self.conv2 = nn.Sequential(*[AGNNConv(requires_grad=True)]*(self.k-1))
        if gnn_type==14:
            self.conv1 = ARMAConv(in_channels=self.in_channels, out_channels=out_channels, num_stacks=1, num_layers=1, 
                                  shared_weights=False, act=F.relu, dropout=0.5, bias=True)
            self.conv2 = nn.Sequential(*[ARMAConv(in_channels=out_channels, out_channels=out_channels, num_stacks=1, num_layers=1, 
                                  shared_weights=False, act=F.relu, dropout=0.5, bias=True)]*(self.k-1))
        if gnn_type==15:
            self.conv1 = SGConv(in_channels=self.in_channels, out_channels=out_channels, K=1, cached=True, bias=True)
            self.conv2 = nn.Sequential(*[SGConv(in_channels=out_channels, out_channels=out_channels, K=1, cached=True, bias=True)]*(self.k-1))
        if gnn_type==16:
            self.conv1 = SGConv(in_channels=self.in_channels, out_channels=out_channels, K=2, cached=True, bias=True)
            self.conv2 = nn.Sequential(*[SGConv(in_channels=out_channels, out_channels=out_channels, K=2, cached=True, bias=True)]*(self.k-1))
        if gnn_type==17:
            self.conv1 = APPNP(K=1, alpha=0.2, bias=True)
            self.conv2 = nn.Sequential(*[APPNP(K=1, alpha=0.2, bias=True)]*(self.k-1))
        if gnn_type==18:
            self.conv1 = APPNP(K=3, alpha=0.2, bias=True)
            self.conv2 = nn.Sequential(*[APPNP(K=3, alpha=0.2, bias=True)]* (self.k-1))
        if gnn_type==19:
            self.conv1 = RGCNConv(in_channels=self.in_channels, out_channels=out_channels, num_relations=3, num_bases=2, bias=True)
            self.conv2 = nn.Sequential(*[RGCNConv(in_channels=out_channels, out_channels=out_channels, num_relations=3, num_bases=2, bias=True)]*(self.k-1))
        if gnn_type==20:
            self.conv1 = SGConv(in_channels=self.in_channels, out_channels=out_channels, K=3, cached=True, bias=True)
            self.conv2 = nn.Sequential(*[SGConv(in_channels=out_channels, out_channels=out_channels, K=3, cached=True, bias=True)]*(self.k-1))
        


# =============================================================================
#         if gnn_type==21:
#             self.conv1 = SignedConv(in_channels=1, out_channels=out_channels, first_aggr=False, bias=True)
#             self.conv2 = SignedConv(in_channels=out_channels, out_channels=out_channels, first_aggr=False, bias=True)
#         if gnn_type==22:
#             self.conv1 = GMMConv(in_channels=1, out_channels=out_channels, dim=2, kernel_size=3, bias=True)
#             self.conv2 = GMMConv(in_channels=out_channels, out_channels=out_channels, dim=2, kernel_size=3, bias=True)
#         if gnn_type==23:
#             self.conv1 = GMMConv(in_channels=1, out_channels=out_channels, dim=5, kernel_size=3, bias=True)
#             self.conv2 = GMMConv(in_channels=out_channels, out_channels=out_channels, dim=5, kernel_size=3, bias=True)
#         if gnn_type==24:
#             self.conv1 = GMMConv(in_channels=1, out_channels=out_channels, dim=2, kernel_size=3, bias=True)
#             self.conv2 = GMMConv(in_channels=out_channels, out_channels=out_channels, dim=2, kernel_size=3, bias=True)
# =============================================================================
        if gnn_type==25:
            self.conv1 = SplineConv(in_channels=self.in_channels, out_channels=out_channels, dim=2, kernel_size=3, is_open_spline=True, 
                                    degree=1, norm=True, root_weight=True, bias=True)
            self.conv2 = nn.Sequential(*[SplineConv(in_channels=out_channels, out_channels=out_channels, dim=2, kernel_size=3, is_open_spline=True, 
                                    degree=1, norm=True, root_weight=True, bias=True)]*(self.k-1))
        if gnn_type==26:
            self.conv1 = SplineConv(in_channels=self.in_channels, out_channels=out_channels, dim=3, kernel_size=3, is_open_spline=False, 
                                    degree=1, norm=True, root_weight=True, bias=True)
            self.conv2 = nn.Sequential(*[SplineConv(in_channels=out_channels, out_channels=out_channels, dim=3, kernel_size=3, is_open_spline=False, 
                                    degree=1, norm=True, root_weight=True, bias=True)]*(self.k-1))
        if gnn_type==27:
            self.conv1 = SplineConv(in_channels=self.in_channels, out_channels=out_channels, dim=3, kernel_size=6, is_open_spline=True, 
                                    degree=1, norm=True, root_weight=True, bias=True)
            self.conv2 = nn.Sequential(*[SplineConv(in_channels=out_channels, out_channels=out_channels, dim=3, kernel_size=6, is_open_spline=True, 
                                    degree=1, norm=True, root_weight=True, bias=True)]*(self.k-1))
        if gnn_type==28:
            self.conv1 = SplineConv(in_channels=self.in_channels, out_channels=out_channels, dim=3, kernel_size=3, is_open_spline=True, 
                                    degree=3, norm=True, root_weight=True, bias=True)
            self.conv2 = nn.Sequential(*[SplineConv(in_channels=out_channels, out_channels=out_channels, dim=3, kernel_size=3, is_open_spline=True, 
                                    degree=3, norm=True, root_weight=True, bias=True)]*(self.k-1))
        if gnn_type==29:
            self.conv1 = SplineConv(in_channels=self.in_channels, out_channels=out_channels, dim=3, kernel_size=6, is_open_spline=True, 
                                    degree=3, norm=True, root_weight=True, bias=True)
            self.conv2 = nn.Sequential(*[SplineConv(in_channels=out_channels, out_channels=out_channels, dim=3, kernel_size=6, is_open_spline=True, 
                                    degree=3, norm=True, root_weight=True, bias=True)]*(self.k-1))

    #def bn(self, i, x):
        #batch_size, num_nodes, num_channels = x.size()

    #    x = x.view(-1, num_channels)
    #    x = getattr(self, 'bn{}'.format(i))(x)
    #    x = x.view(batch_size, num_nodes, num_channels)
    #    return x
    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x
    
    def forward(self, x, edge_index=None,edge_weight=None,adj=None, mask=None):
        #print('forward gnn')
        #print(self.in_channels)
        #print(self.conv1.in_channels)
        #print('end forward gnn')
        #print(adj.size())
        if self.gnn_type<=1:
            #batch_size, num_nodes, in_channels = x.size()
        #batch_size,num_nodes,num_nodes=adj.size()
            batch_size, num_nodes, _ = x.size()
            
            if self.k >1 :
                x = self.conv1(x, adj)
                 
                #x = x.view(-1,x.size()[-1])#x.size()[-1]=out_channels
                x = x.permute(0,2,1) #new version

                x = self.activ(self.bn1(x))
                x = x.permute(0,2,1) #new version
                if self.jump==None:
                    for i in range(self.k-1):
                        x = x.view(batch_size, num_nodes, -1)
                        #print('adj')
                        #print(adj.size())
                        x = self.conv2[i](x, adj)
                        #x = x.view(-1,x.size()[-1])
                        x = x.permute(0,2,1) #new version
                        x = self.activ(self.bn2[i](x))
                        x = x.permute(0,2,1) #new version
                else :
                    if self.jump=='max':
                        xs=[x]
                        for i in range(self.k-1):
                            #x = x.view(batch_size, num_nodes, -1) #old_version
                            x = self.conv2[i](x, adj)
                            #x = x.view(-1,x.size()[-1]) #old_version
                            x = x.permute(0,2,1) #new version
                            x = self.activ(self.bn2[i](x))
                            x = x.permute(0,2,1) #new version
                            xs = xs + [x]
                        
                        x = self.jk(xs)
                    else:#jump=='lstm'
                        print('debug1')
                        print(x.size())
                        print(self.out_channels)
                        xs=[x.contiguous().view(-1,self.out_channels)]#batch*num_nodes,out_channels
                        for i in range(self.k-1):
                            #x = x.view(batch_size, num_nodes, -1) #old_version
                            x = self.conv2[i](x, adj)
                            #x = x.view(-1,x.size()[-1]) #old_version
                            x = x.permute(0,2,1) #new version
                            x = self.activ(self.bn2[i](x))
                            x = x.permute(0,2,1) #new version
                            xs = xs + [x.contiguous().view(-1,self.out_channels)]
                        
                        x = self.jk(xs).view(batch_size,num_nodes,self.out_channels)
                
                        
            else:
                x = self.conv1(x, adj)
                x = x.permute(0,2,1) #new version
                #x = x.view(-1,x.size()[-1]) #old_version
                x = self.activ(self.bn1(x))
                x = x.permute(0,2,1) #new version
            #x = x.view(batch_size, num_nodes, -1)
             
        #GAT,AGNN,GraphConv, only accepts edge_index   [6,9,10,11,12]  
        elif self.gnn_type in [2,3,4,5,6,7,8,9,10,11,12,14,15,16] and edge_weight is None:
            if self.k >1 :
                
                if self.jump==None:
                    x = self.activ(self.bn1( self.conv1(x, edge_index)))
                    for i in range(self.k-1):
                        x = self.activ(self.bn2[i]( self.conv2[i](x, edge_index)))
                else:
                    
                    x = self.activ(self.bn1( self.conv1(x, edge_index)))
                    xs = [x]
                    for i in range(self.k-1):
                        x = self.activ(self.bn2[i]( self.conv2[i](x, edge_index)))
                        xs = xs + [x]
                    
                    x = self.jk(xs)
                
            else:
                x = self.activ(self.bn1( self.conv1(x, edge_index)))
                
        elif self.gnn_type in [2,3,4,5,7,8,14,15,16,17,18] and not (edge_weight is None):
            if self.k >1 :
                
                if self.jump==None:
                    x = self.activ(self.bn1( self.conv1(x, edge_index,edge_weight)))
                    for i in range(self.k-1):
                        x = self.activ(self.bn2[i]( self.conv2[i](x, edge_index,edge_weight)))
                else:
                    
                    x = self.activ(self.bn1( self.conv1(x, edge_index,edge_weight)))
                    xs = [x]
                    for i in range(self.k-1):
                        x = self.activ(self.bn2[i]( self.conv2[i](x, edge_index,edge_weight)))
                        xs = xs + [x]
                    x = self.jk(xs)
                
            else:
                x = self.activ(self.bn1( self.conv1(x, edge_index,edge_weight)))
            
        else:
            raise ValueError('If edge_weight is not None then gnn_type must be in 2,3,4,5,7,8,14,15,16,17,18')

        return x

def get_num_par(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def train_c(train_loader, y_train, censor_train,mode,x_train=None,adj_train=None,device='gpu',model=None,optimizer=None,H_1=1,debug=False):
    model.train()
    if mode=='edge'or mode=='edge_weight':
        outx=[]
        for data in train_loader:
            data = data.to(device)
            #xdata=data
            #t_x=data.x.type(torch.float)
            #t_edge=data.edge_index
            #t_batch=data.batch
            #break
            optimizer.zero_grad()
            if mode=='edge':
                output = model(data.x.type(torch.float),data.edge_index,data.batch)
                outx.append(output)
            elif mode=='edge_weight':
                output = model(data.x.type(torch.float), data.edge_index,data.edge_attr.type(torch.float))
            
        #c_idx = c_index(output,torch.from_numpy(y_train),torch.from_numpy( censor_train))
        loss = neg_par_log_likelihood(output, y_train, censor_train)#event=0,censored
        loss.backward()
        optimizer.step()
        
        return loss 
    else:
        
        #print(x_train is None)
        optimizer.zero_grad()
         
        #t0=time.time()
        #output,link_loss,ent_loss = model(x=x_train, adj=adj_train,mask=None)
        output = model(x=x_train)
        #t0=time.time() 
        #c_idx = c_index(output,y_train,censor_train)
        #print('cidx time:{}s'.format(time.time()-t0)) 
        #loss = neg_par_log_likelihood(output, y_train, censor_train) + H_1*(  ent_loss)#event=0,censored
        #t0=time.time()
        loss = neg_par_log_likelihood(output, y_train, censor_train)  
        #print('loss time:{}s'.format(time.time()-t0))
        #t0=time.time()
        loss.backward()
        optimizer.step()
        if debug==True:
            print(output)
        #print('backward prop time:{}s'.format(time.time()-t0))
        return loss  
#
#model.gnn1.conv1(data.x.type(torch.float), data.edge_index,data.edge_attr.type(torch.float).view(-1))



def test(test_loader,y_test,censor_test,mode,x_test=None,adj_test=None,device='gpu',model=None,optimizer=None):
    model.eval()
    if mode=='edge'or mode=='edge_weight':
        for data in test_loader:
            data = data.to(device)
            #xdata=data
            if mode=='edge':
                #print(data.x.size())
                pred = model(data.x.type(torch.float), data.edge_index)
            elif mode=='edge_weight':
                pred = model(data.x.type(torch.float), data.edge_index,data.edge_attr.type(torch.float))
            #pred = model(data.x.type(torch.float),  data.edge_index)
            cidx = c_index(pred, y_test, censor_test)
        return cidx
    else:
    
        #optimizer.zero_grad()
        with torch.no_grad():
            t0=time.time()
            #pred ,link_loss,ent_loss= model(x_test, adj=adj_test)
            pred = model(x_test )
            cidx = c_index(pred, y_test,censor_test)
            #print('test mode time:{}s'.format(time.time()-t0))
        return cidx

#debug
# =============================================================================
#         model(xdata.x.type(torch.float), xdata.edge_index)
# model.gnn1.conv1(xdata.x.type(torch.float), xdata.edge_index)    
# =============================================================================
#

def do_c(maxit, y_train, censor_train, y_test, censor_test,mode,train_loader,test_loader,x_train,x_test,adj_train,adj_test,optimizer,device,model=None):
    if x_train is not None:
        x_train = x_train.type(torch.float32).to(device)
        x_test = x_test.type(torch.float32).to(device)
   
    y_train = torch.from_numpy(y_train).type(torch.float32).to(device)
    censor_train=torch.from_numpy( censor_train).type(torch.long).to(device)
    y_test = torch.from_numpy(y_test).type(torch.float32).to(device)
    
    censor_test=torch.from_numpy( censor_test).type(torch.long).to(device)
    if adj_train is not None:
        for i in range(len(adj_train)):
            adj_train[i]=adj_train[i].type(torch.float32).to(device)
    
        for i in range(len(adj_test)):
            adj_test[i]=adj_test[i].type(torch.float32).to(device)
    h_debug=False
    for epoch in range(1, maxit):
        temp_loss=train_c(train_loader, y_train, censor_train,mode,x_train,adj_train,device,model,optimizer,debug=h_debug)
        #print('training at epoch {} starts.'.format(epoch))
        
        if epoch %100==0:
            test_cidx = test(test_loader, y_test, censor_test,mode,x_test,adj_test,device,model,optimizer)
            print('epock {}, trloss = {}, test cidx = {}'.format(epoch,temp_loss.cpu().data.numpy().ravel()[0],test_cidx))
            if epoch>=2200:
                print(temp_loss)
                h_debug=True
    test_cidx = test(test_loader, y_test, censor_test,mode,x_test,adj_test,device,model,optimizer)
    print('final testing_c_index = {}'.format(test_cidx))
    return test_cidx
#########################################################################
# =============================================================================
# from time import time
# mode=process_type
# t0=time()
# model.train()
# train_c(train_loader, y_train, censor_train,mode,x_train,adj_train)
# t1=time()-t0 
# 
# print(t1)
# #model.eval()  
# #with torch.no_grad():         
# #    pred,link_loss,ent_loss= model(x_test, adj=adj_test.type(torch.float32))
# #t2=time()-t0
# 
# print('t1={}, t2={}'.format(t1,t2))    
# =============================================================================
###########test time

 

class block_edge_conv(torch.nn.Module):#gcn
    def __init__(self,num_features,in_channels=1,out_channels=4,gnn_k=1,gnn_type=2,activation='leaky',res=False,jump=None,reduce=True):
        super(block_edge_conv, self).__init__()
        
        self.num_features = num_features    
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.gnn_k=gnn_k
        self.gnn_type=gnn_type
        self.reduce=reduce
        if activation=='leaky':
            self.activ=F.leaky_relu
        elif activation=='elu':
            self.activ=F.elu
        elif activation=='relu':
            self.activ=F.relu
        if jump=='lstm' and out_channels%2==1 and gnn_k%2==1:
            raise ValueError('out_channels%2==1 and gnn_k%2==1')
        self.gnn1= GNN(self.in_channels, self.out_channels, gnn_type=self.gnn_type,gnn_k=self.gnn_k,add_loop=False,jump=jump)
         
        self.lin1times1 = torch.nn.Linear(self.out_channels,1)
        
    def forward(self, x, edge_index,edge_weight=None):
        tempsize = x.size()
        print('blockedgeinput size')
        print(tempsize)
        x = self.gnn1(x, edge_index,edge_weight)
        print('gnn1 outputsize in block')
        print(x.size())
        #x = x.view(-1,self.num_features,self.out_channels)
        
        #x = x.view(tempsize)
        if self.reduce==True:
            x = self.activ(self.lin1times1(x))
            x = x.view(tempsize)
        return x




class fc_edge(torch.nn.Module):#gcn
    def __init__(self,num_features,extra_nodelist=[1],activation='leaky' ):
        super(fc_edge, self).__init__()
        
        self.num_features = num_features    
        
        if activation=='leaky':
            self.activ=[nn.LeakyReLU()]*len(extra_nodelist)
        elif activation=='elu':
            self.activ=[nn.ELU()]*len(extra_nodelist)
        elif activation=='relu':
            self.activ=[nn.ReLU()]*len(extra_nodelist)
         
        if len(extra_nodelist)==1:
               
            self.lin1 = torch.nn.Linear(self.num_features, 1) 
        else:
            #print(len(nodelist))
            li=[torch.nn.Linear(self.num_features, extra_nodelist[0]),torch.nn.BatchNorm1d(extra_nodelist[0]),\
                self.activ[0],nn.Dropout()]
            for i in range(1,len(extra_nodelist)-1): 
                li = li+[torch.nn.Linear(extra_nodelist[i-1], extra_nodelist[i]),torch.nn.BatchNorm1d(extra_nodelist[i]),\
                self.activ[i],nn.Dropout()]
            li = li + [torch.nn.Linear(extra_nodelist[-2], 1)]
            self.lin1 = nn.Sequential(*li)
        
        
    def forward(self, x ):
        
        x = x.view(x.size()[0:2])
        x = self.lin1(x)
        #x = self.lin2(x)
        return x



class block_adj_embed(torch.nn.Module):
    #note that for gnn2_pool,the input dimension is the number of hidden nodes after the first pooling operation
    def __init__(self,num_features,in_channels=1,out_channels=50,gnn_k=1,gnn_type=1,activation='leaky',res=False,jump=None\
                 ,reduce=True):
        super(block_adj_embed, self).__init__()
        self.num_features = num_features    
        self.in_channels=in_channels 
        self.out_channels=out_channels
        self.reduce=reduce
        
        if jump=='lstm' and out_channels%2==1 and gnn_k%2==1:
            raise ValueError('out_channels%2==1 and gnn_k%2==1') 
        
        if gnn_type>=2:
            raise ValueError('gnn_type must be 0 or 1') 
        if activation=='leaky':
            self.activ=F.leaky_relu
        elif activation=='elu':
            self.activ=F.elu
        elif activation=='relu':
            self.activ=F.relu
            
        self.gnn0_embed = GNN(self.in_channels, out_channels,gnn_k=gnn_k,gnn_type=gnn_type,add_loop=False,jump=jump)
        self.lin0_1times1 = torch.nn.Linear(self.out_channels,in_channels)
        
    def forward(self, x_input, adj, mask=None):
        print('input')
        print(x_input.size())
        print(adj.size())
        #print(self.gnn0_embed)
        x = self.gnn0_embed(x_input,adj=adj,mask=mask)
        #print('after gnn0_embed')
        #print(x.size())
        if self.reduce==True:
            x = self.lin0_1times1(x)
        return x

    
class block_adj_pool(torch.nn.Module):
    #note that for gnn2_pool,the input dimension is the number of hidden nodes after the first pooling operation
    def __init__(self,num_features,in_channels=1,out_channels=10,out_clusters=20,gnn_k=1,gnn_type=1,activation='leaky',res=False,jump=None\
                 ,shrink_factor=0.2):
        super(block_adj_pool, self).__init__()
        self.num_features = num_features    
        self.in_channels=in_channels
        self.out_clusters=out_clusters
        self.out_channels=out_channels
        self.sh_f=shrink_factor
       
        assert out_clusters*shrink_factor>=2
        if gnn_type>=2:
            raise ValueError('gnn_type must be 0 or 1') 
        if activation=='leaky':
            self.activ=F.leaky_relu
        elif activation=='elu':
            self.activ=F.elu
        elif activation=='relu':
            self.activ=F.relu
        if jump=='lstm' and out_channels%2==1 and gnn_k%2==1:
            raise ValueError('out_channels%2==1 and gnn_k%2==1')
        self.gnn1_pool = GNN(self.in_channels, out_clusters,gnn_k=gnn_k,gnn_type=gnn_type,add_loop=False,jump=jump)
        self.gnn1_embed = GNN(self.in_channels, out_channels,gnn_k=gnn_k,gnn_type=gnn_type,add_loop=False,jump=jump)

    def forward(self, x_input, adj, mask=None):
        #print('forward diff')
        s = self.gnn1_pool(x_input, adj=adj, mask=mask)
         
        x = self.gnn1_embed(x_input, adj=adj, mask=mask)
         
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        
        return x, l1, e1, adj 


#test block code
        
# =============================================================================
# x_edge_train=xdata_edge.x.type(torch.float)
# edge_index_train=xdata_edge.edge_index    
#  
# x_weight=xdata_weight.x.type(torch.float)
# x_weight_index=xdata_weight.edge_index
# x_weight_weight=xdata_weight.edge_attr.type(torch.float)
# x_adj=x_train
# x_adj_adj=adj_train
# #[2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18]
# a1=block_edge_conv(num_features,gnn_k=3,gnn_type=3,activation='leaky',jump='lstm')
# a1(x_edge_train,edge_index_train).size()
# a2=block_edge_conv(num_features,gnn_k=3,gnn_type=2,activation='elu',jump='lstm')
# a2(x_edge_train,edge_index_train).size()
# a2(x_weight,x_weight_index,x_weight_weight).size()
# 
# a3=block_adj_embed(num_features,gnn_k=3,gnn_type=1,activation='leaky',jump='lstm')
# a3(x_adj,x_adj_adj).size()
# 
# a4=block_adj_pool(num_features,gnn_k=3,gnn_type=1,activation='leaky',jump='lstm')
# x4,l4,e4,adj4=a4(x_adj,x_adj_adj)
# a5=fc_edge(88,[8,1])
# a5(torch.ones(5,88))
# =============================================================================
#edge x.size()=batch*num_fetures,channels


#adj x.size()=batch,num_features,n_channels


class cell0(torch.nn.Module):#gcn
    #multiple edge_block,with customized depth
    def __init__(self,num_features,in_channels=1,out_channels=4,gnn_k=1,gnn_type=2,activation='leaky',res=False,jump=None,depthlist=[1],nb=1,nodelist=[1],fc_exist=False,reduce_tail=True):
        super(cell0, self).__init__()
         #gnn_type 15 ,16,20 corresponds to A,A**2,A**3,where A is the adjacency matrix
        self.num_features = num_features    
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.gnn_k=gnn_k
        self.gnn_type=gnn_type
        self.res = res
        self.reduce_tail=reduce_tail
        self.fc=fc_edge(self.num_features , nodelist)
        self.nb=nb
        self.conv=[]
        self.depthlist=depthlist
        self.fc_exist=fc_exist
        if activation=='leaky':
            self.activ=F.leaky_relu
        elif activation=='elu':
            self.activ=F.elu
        elif activation=='relu':
            self.activ=F.relu
        self.conv=nn.ModuleList([])
        for j in range(nb):    
            self.conv.append(block_edge_conv(num_features,in_channels=self.in_channels,out_channels=self.out_channels,\
                                       gnn_k=gnn_k,gnn_type=gnn_type,activation=activation,jump=jump))
            for i in range(depthlist[j]-1):
                self.conv.append(block_edge_conv(num_features,in_channels=self.in_channels,out_channels=self.out_channels,\
                                       gnn_k=gnn_k,gnn_type=gnn_type,activation=activation,jump=jump))
            
            self.conv[-1].reduce=reduce_tail
         
        
    def forward(self, x, edge_index,adj=None):
        #the input size is (num_features*batch,1)
        print('cell0input')
        print(x.size())
        output=[]
        if self.res==False:
            for j in range(self.nb):
                for i in range(self.depthlist[j]):
                    x = self.conv[j*self.nb+i](x, edge_index,adj)
                output.append(x)
        else:
            for j in range(self.nb):
                for i in range(self.depthlist[j]):
                    x0 = x 
                    x = self.conv[j*self.nb+i](x, edge_index,adj)
                    print('after conv')
                    print(x.size())
                    x =x + x0
                output.append(x)
        if  self.nb >1 and self.reduce_tail==True:
            x = torch.cat(tuple(output),dim=-1)
            x = x.mean(dim=-1)#here the size of x is (num_features*batch)
        #the input size is (num_features*batch,1),fc changes it to be batch,1
        elif self.nb > 1 and  self.reduce_tail==False:
            torch.stack(output,dim=0).mean(dim=0)
        if self.fc_exist==True: 
            x = x.view(-1,self.num_features)
            x = self.fc(x)
        
        return x



class cell1(torch.nn.Module):#gcn
    #multiple edge_block,with customized depth
    def __init__(self,num_features,in_channels=1,out_channels=4,gnn_k=1,gnn_type=2,activation='leaky',res=False,jump=None,depthlist=[1],nb=1,nodelist=[1],fc_exist=False,reduce_tail=True):
        super(cell1, self).__init__()
         #gnn_type 15 ,16,20 corresponds to A,A**2,A**3,where A is the adjacency matrix
        self.num_features = num_features    
        
        self.out_channels=out_channels
        self.gnn_k=gnn_k
        self.gnn_type=gnn_type
        self.res = res
        self.in_channels=in_channels
        self.fc=fc_edge(self.num_features , nodelist)
        self.nb=nb
        self.conv=nn.ModuleList([])
        self.depthlist=depthlist
        self.fc_exist=fc_exist
        self.reduce_tail=reduce_tail
        if activation=='leaky':
            self.activ=F.leaky_relu
        elif activation=='elu':
            self.activ=F.elu
        elif activation=='relu':
            self.activ=F.relu
        for j in range(nb):    
            self.conv.append(block_adj_embed(num_features,in_channels=self.in_channels,out_channels=self.out_channels,\
                                       gnn_k=gnn_k,gnn_type=gnn_type, activation=activation,jump=jump))
            for i in range(1,depthlist[j]):
                self.conv.append(block_adj_embed(num_features,in_channels=self.in_channels,out_channels=self.out_channels,\
                                       gnn_k=gnn_k,gnn_type=gnn_type, activation=activation,jump=jump))
            self.conv[-1].reduce=reduce_tail
            

         
        
    def forward(self, x, adj,mask=None):
        #the input size is (batch,num_features,n_channels)
        x_input = x
        output=[]
        if self.res==False:
            for j in range(self.nb):
                print('j={}'.format(j))
                x = x_input
                for i in range(self.depthlist[j]):
                    print('i={}'.format(i))
                    x = self.conv[j*self.nb+i](x, adj[j], mask)
                output.append(x)
        else:
            for j in range(self.nb):
                for i in range(self.depthlist[j]):
                    x0 = x_input 
                    x = self.conv[j*self.nb+i](x, adj[j], mask) + x0
                output.append(x)
         
        if  self.nb >1 and self.reduce_tail==True:
            x = torch.cat(tuple(output),dim=-1).mean(dim=-1)
            #x = x.mean(dim=-1)#here the size of x is (num_features*batch)
        #the input size is (num_features*batch,1),fc changes it to be batch,1
        elif self.nb > 1 and  self.reduce_tail==False:
            torch.stack(output,dim=0).mean(dim=0)
        if self.fc_exist==True: 
            x = x.view(-1,self.num_features)
            x = self.fc(x)
        else:
             
            x = x.view(*(list(x.size()[0:2])+[-1]))
            
        return x



    
class Netadj(torch.nn.Module):
    #note that for gnn2_pool,the input dimension is the number of hidden nodes after the first pooling operation
    def __init__(self,num_features ,out_channels=4,out_clusters_list=[40,20],gnn_k=2,gnn_type=1,\
                 activation='leaky',res=False,jump=None,depthlist=[2,2],fc_node_list=[20,10,1],dimred=True):
        super(Netadj, self).__init__()
        if len(depthlist)>3:
            raise ValueError('len(depthlist) must be smaller or equal to 3.')
        if gnn_k==1 and ( (jump is not None) or (res==True) ):
            raise ValueError('if gnn_k==1 then jump must be None and res must be False.')
        if out_channels*gnn_k%2!=0 and jump=='lstm':
            raise ValueError('out_channels*gnn_k should be even numbers if jump==\'lstm\'.')
        self.num_features = num_features    
        self.out_clusters_list=out_clusters_list
        self.out_channels=out_channels
        self.dimred = dimred
        #out_clusters_list,the size of the num_features or number of the clusters
        self.embed_list=nn.ModuleList([])
        self.pool_list=nn.ModuleList([])
        if dimred ==True:
         
            self.embed_list.append(cell1(num_features,in_channels=1,out_channels=self.out_channels,\
                gnn_type=gnn_type,gnn_k=gnn_k,res=res,jump=jump,depthlist=depthlist,nb=len(depthlist),fc_exist=False,reduce_tail=True))
            for i in range(1,len(out_clusters_list)):
                self.embed_list=self.embed_list.append(cell1(self.out_clusters_list[i-1],in_channels=1,out_channels=self.out_channels,\
                gnn_type=gnn_type,gnn_k=gnn_k,res=res,jump=jump,depthlist=depthlist,nb=len(depthlist),fc_exist=False,reduce_tail=True) )
             
            self.pool_list.append(cell1(num_features,in_channels=1,out_channels=out_clusters_list[0],\
                gnn_type=gnn_type,gnn_k=gnn_k,res=res,jump=jump,depthlist=[1],nb=1,fc_exist=False,reduce_tail=False))
            for i in  range(1,len(self.out_clusters_list)):
                self.pool_list=self.pool_list.append(cell1(self.out_clusters_list[i-1],in_channels=1,out_channels=self.out_clusters_list[i],\
                gnn_type=gnn_type,gnn_k=gnn_k,res=res,jump=jump,depthlist=[1],nb=1,fc_exist=False,reduce_tail=False))
             
            
            
            self.postfc= fc_edge(out_clusters_list[-1],extra_nodelist=fc_node_list)
            self.lin1times1 = torch.nn.Linear(self.out_channels,1)
        else:#dimred ==False:
            embed_list=[cell1(num_features,in_channels=1,out_channels=self.out_channels,\
                gnn_type=gnn_type,gnn_k=gnn_k,res=res,jump=jump,depthlist=depthlist,nb=len(depthlist),fc_exist=False,reduce_tail=False) ]
            for i in range(1,len(out_clusters_list)):
                embed_list=embed_list+[ cell1(self.out_clusters_list[i-1],in_channels=self.out_channels,out_channels=self.out_channels,\
                gnn_type=gnn_type,gnn_k=gnn_k,res=res,jump=jump,depthlist=depthlist,nb=len(depthlist),fc_exist=False,reduce_tail=False) ]
             
            self.pool_list.append(cell1(num_features,in_channels=1,out_channels=out_clusters_list[0],\
                gnn_type=gnn_type,gnn_k=gnn_k,res=res,jump=jump,depthlist=[1],nb=1,fc_exist=False,reduce_tail=False))
            for i in  range(1,len(self.out_clusters_list)):
                self.pool_list.append(cell1(self.out_clusters_list[i-1],in_channels=self.out_channels,out_channels=self.out_clusters_list[i],\
                gnn_type=gnn_type,gnn_k=gnn_k,res=res,jump=jump,depthlist=[1],nb=1,fc_exist=False,reduce_tail=False))
             
            
            self.postfc= fc_edge(out_clusters_list[-1],extra_nodelist=fc_node_list)
            self.lin1times1 = torch.nn.Linear(self.out_channels,1)
        
    

    def forward(self, x, adj, mask=None):
        #print('forward diff')
        l_loss=0
        e_loss=0
        for i in range(len(self.out_clusters_list)):
            print(x.size())    
            print('layer')
            print(i)
            #print(x.size())
            #print('in forward')
            #print(x.device)
            #print(adj[0].device)
            s = self.pool_list[i](x, adj=adj, mask=mask) #batch,num_features,out_clusteres_list
            #print('pool finished')
            x = self.embed_list[i](x, adj=adj, mask=mask)#batch,num_features,out_channels
            #print('before diff pool')
            #print(s.size())
            #this is correct since we only first order adj
            x, adj, l1, e1 = dense_diff_pool(x, adj[0], s, mask)
            #adj=[adj[0],adj[0]**2,adj[0]**3]
            #print('after diff pool')
            #print(x.size())
            l_loss+=l1
            e_loss+=e1
# =============================================================================
#         print('x')
#         print(x.size())
#         print('end x.size')
#         print('adj')
#         print(adj.size())
#         print('adj end')
#         print('s size')
#         print(s.size())
#         print('s size end')
# =============================================================================
        if self.dimred==False:
            x = self.lin1times1(x)
         
        x = x.view(x.size()[0:2])
         
        x = self.postfc(x)
        return x, l_loss, e_loss



    
class Net0(torch.nn.Module):
    #note that for gnn2_pool,the input dimension is the number of hidden nodes after the first pooling operation
    def __init__(self,num_features ,out_channels=4,out_clusters_list=[40,20],gnn_k=2,gnn_type=1,\
                 activation='leaky',res=False,jump=None,depthlist=[2,2],fc_node_list=[20,20,1],dimred=True):
        super(Net0, self).__init__()
        if len(depthlist)>3:
            raise ValueError('len(depthlist) must be smaller or equal to 3.')
        if gnn_k==1 and ( (jump is not None) or (res==True) ):
            raise ValueError('if gnn_k==1 then jump must be None and res must be False.')
        if out_channels*gnn_k%2!=0 and jump=='lstm':
            raise ValueError('out_channels*gnn_k should be even numbers if jump==\'lstm\'.')
        self.num_features = num_features    
        self.out_clusters_list=out_clusters_list
        self.out_channels=out_channels
        self.dimred = dimred
        #out_clusters_list,the size of the num_features or number of the clusters
        self.embed_list=nn.ModuleList([])
        self.pool_list=nn.ModuleList([])
        if dimred ==True:
         
            self.embed_list.append(cell0(num_features,in_channels=1,out_channels=self.out_channels,\
                gnn_type=gnn_type,gnn_k=gnn_k,res=res,jump=jump,depthlist=depthlist,nb=len(depthlist),fc_exist=False,reduce_tail=True))
            for i in range(1,len(out_clusters_list)):
                self.embed_list=self.embed_list.append(cell0(self.out_clusters_list[i-1],in_channels=1,out_channels=self.out_channels,\
                gnn_type=gnn_type,gnn_k=gnn_k,res=res,jump=jump,depthlist=depthlist,nb=len(depthlist),fc_exist=False,reduce_tail=True) )
             
            self.pool_list.append(cell0(num_features,in_channels=1,out_channels=out_clusters_list[0],\
                gnn_type=gnn_type,gnn_k=gnn_k,res=res,jump=jump,depthlist=[1],nb=1,fc_exist=False,reduce_tail=False))
            for i in  range(1,len(self.out_clusters_list)):
                self.pool_list=self.pool_list.append(cell0(self.out_clusters_list[i-1],in_channels=1,out_channels=self.out_clusters_list[i],\
                gnn_type=gnn_type,gnn_k=gnn_k,res=res,jump=jump,depthlist=[1],nb=1,fc_exist=False,reduce_tail=False))
             
            
            
            self.postfc= fc_edge(out_clusters_list[-1],extra_nodelist=fc_node_list)
            self.lin1times1 = torch.nn.Linear(self.out_channels,1)
        else:#dimred ==False:
            embed_list=[cell0(num_features,in_channels=1,out_channels=self.out_channels,\
                gnn_type=gnn_type,gnn_k=gnn_k,res=res,jump=jump,depthlist=depthlist,nb=len(depthlist),fc_exist=False,reduce_tail=False) ]
            for i in range(1,len(out_clusters_list)):
                embed_list=embed_list+[ cell0(self.out_clusters_list[i-1],in_channels=self.out_channels,out_channels=self.out_channels,\
                gnn_type=gnn_type,gnn_k=gnn_k,res=res,jump=jump,depthlist=depthlist,nb=len(depthlist),fc_exist=False,reduce_tail=False) ]
             
            self.pool_list.append(cell0(num_features,in_channels=1,out_channels=out_clusters_list[0],\
                gnn_type=gnn_type,gnn_k=gnn_k,res=res,jump=jump,depthlist=[1],nb=1,fc_exist=False,reduce_tail=False))
            for i in  range(1,len(self.out_clusters_list)):
                self.pool_list.append(cell0(self.out_clusters_list[i-1],in_channels=self.out_channels,out_channels=self.out_clusters_list[i],\
                gnn_type=gnn_type,gnn_k=gnn_k,res=res,jump=jump,depthlist=[1],nb=1,fc_exist=False,reduce_tail=False))
             
            
            self.postfc= fc_edge(out_clusters_list[-1],extra_nodelist=fc_node_list)
            self.lin1times1 = torch.nn.Linear(self.out_channels,1)
        
    

    def forward(self, x, edge,adj):
        #print('forward diff')
        l_loss=0
        e_loss=0
        for i in range(len(self.out_clusters_list)):
            print(x.size())    
            print('layer')
            print(i)
            #print(x.size())
            #print('in forward')
            #print(x.device)
            #print(adj[0].device)
            s = self.pool_list[i](x, edge) #batch,num_features,out_clusteres_list
            #print('pool finished')
            x = self.embed_list[i](x, edge)#batch,num_features,out_channels
            #print('before diff pool')
            #print(s.size())
            print('before dense_diff_pool')
            print(s.size())
            print(x.size())
            x, adj, l1, e1 = dense_diff_pool(x, adj[0], s, mask)
            #adj=[adj[0],adj[0]**2,adj[0]**3]
            #print('after diff pool')
            #print(x.size())
            l_loss+=l1
            e_loss+=e1
# =============================================================================
#         print('x')
#         print(x.size())
#         print('end x.size')
#         print('adj')
#         print(adj.size())
#         print('adj end')
#         print('s size')
#         print(s.size())
#         print('s size end')
# =============================================================================
        if self.dimred==False:
            x = self.lin1times1(x)
         
        x = x.view(x.size()[0:2])
         
        x = self.postfc(x)
        return x, l_loss, e_loss



    
class Netglobal(torch.nn.Module):
    #note that for gnn2_pool,the input dimension is the number of hidden nodes after the first pooling operation
    def __init__(self,num_features ,out_channels=4,out_clusters_list=[1],gnn_k=2,gnn_type=1,\
                 activation='leaky',res=False,jump=None,depthlist=[2],fc_node_list=[20,20,1],global_pool=0):
        super(Netglobal, self).__init__()
        if len(depthlist)>3:
            raise ValueError('len(depthlist) must be smaller or equal to 3.')
        if gnn_k==1 and ( (jump is not None) or (res==True) ):
            raise ValueError('if gnn_k==1 then jump must be None and res must be False.')
        if out_channels*gnn_k%2!=0 and jump=='lstm':
            raise ValueError('out_channels*gnn_k should be even numbers if jump==\'lstm\'.')
        self.num_features = num_features    
        self.out_clusters_list=out_clusters_list
        self.out_channels=out_channels
        self.global_pool = global_pool
        
        #out_clusters_list,the size of the num_features or number of the clusters
        self.embed_list=nn.ModuleList([])
        #self.pool_list=nn.ModuleList([])
        self.embed_list.append(cell0(num_features,in_channels=1,out_channels=self.out_channels,\
                gnn_type=gnn_type,gnn_k=gnn_k,res=res,jump=jump,depthlist=depthlist,nb=len(depthlist),fc_exist=False,reduce_tail=False))
        
            
        self.postfc= fc_edge(out_channels,extra_nodelist=fc_node_list)
        #self.lin1times1 = torch.nn.Linear(self.out_channels,1)
        
    

    def forward(self, x, edge,batch):
         
       
             
        x = self.embed_list[0](x, edge)#batch,num_features,out_channels
             
            #x, adj, l1, e1 = dense_diff_pool(x, adj[0], s, mask)
             
            #l_loss+=l1
            #e_loss+=e1
        #0 mean, 1 max
        print('befpre global x.size')
        print(x.size())
        if self.global_pool==0:
            x = torch.stack([global_mean_pool(x[:,i],batch) for i in range(x.size()[-1])],dim=-1)
        else:
            x = torch.stack([global_max_pool(x[:,i],batch) for i in range(x.size()[-1])],dim=-1)
        print('after global')
        print(x.size())
        #x = x.view(x.size()[0:2])
         
        x = self.postfc(x)
        return x 
    
   
class Netfc(torch.nn.Module):
    #note that for gnn2_pool,the input dimension is the number of hidden nodes after the first pooling operation
    def __init__(self,num_features ,fc_node_list=[50,1]):
        super(Netfc, self).__init__()
        
        self.num_features = num_features    
        
       
        self.postfc= fc_edge(self.num_features,extra_nodelist=fc_node_list)
        #self.lin1times1 = torch.nn.Linear(self.out_channels,1)
        
    

    def forward(self, x):
       
        #print(x.size())
        x = self.postfc(x)
        return x 
#data_num=int(sys.argv[1])
#process_type=['edge','edge_weight','adj'][int(sys.argv[2])]
#epoch=int(sys.argv[3])
#n_channels=int(sys.argv[4])
#r_shrink=double(sys.argv[5])
#jump_type=[None,'max','lstm'][int(sys.argv[6])]
#gnn_type=int(sys.argv[7])
#gnn_k=int(sys.argv[8])
#survpath=sys.argv[7]
 
#data_num=5
#process_type='edge'
#process_type='edge_weight'
#process_type='adj'
#epoch=500
#n_channels=3

#gnn_type=1
#gnn_k=3
#jump_type='max'
#sh_f=0.2
#survpath='C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer'

def experiment(data_num,process_type,epoch,n_channels,jump_type,gnn_type,gnn_k,survpath,node_list=[100,1]):
    t0=time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device=('cpu')
    if process_type!='adj':
        data_list,y,censor,trainset,testset,num_features,adj_train,adj_test = get_data_edge_lung(survpath,data_num,data_type=process_type,device=device)
    else:
        data_list,y,censor,trainset,testset,num_features,adj_train,adj_test = get_data_edge_lung(survpath,data_num,data_type=process_type,device=device)
    
    if torch.cuda.is_available():
        print(torch.cuda.memory_allocated(device=device))
    
    
    train_list=[data_list[i] for i in trainset]
    test_list=[data_list[i] for i in testset]
    y_train=np.array([y[i] for i in trainset])
    y_test=np.array([y[i] for i in testset])
    censor_train=np.array([censor[i] for i in trainset])
    censor_test=np.array([censor[i] for i in testset])
    
    
    #Global
    if process_type!='adj':
        train_loader=DataLoader(train_list,batch_size=len(train_list),shuffle=False)
        test_loader=DataLoader(test_list,batch_size=len(test_list),shuffle=False)
        #train_loader=DataLoader(train_list,batch_size=10,shuffle=False)
        #test_loader=DataLoader(test_list,batch_size=10,shuffle=False)
        x_train=None
        x_test=None
        
    else:
        train_loader = None
        test_loader = None
        x_train=data_list[trainset]
        x_test=data_list[testset]
        
        
    
    if process_type!='adj':
    #gnn_type=2-3GCONV,9-14GAT,4-5Cheb,7-8GatedGraph,6GraphConv,15SG,14ARMA #6,9,10,11,12 only edge
        model = Netglobal(num_features,out_channels=n_channels,gnn_type=gnn_type,gnn_k=gnn_k,fc_node_list=[30,1],res=True,jump=jump_type).to(device)
    else:
        model = Netfc(num_features,fc_node_list=node_list).type(torch.float32).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) 
    #print('before do_c')
    #print(list(model.pool_list[0].conv[0].parameters())[0].device)
    print('loading and prepro takes {}s'.format(time.time()-t0))
    ret=do_c(epoch, y_train, censor_train, y_test, censor_test,process_type,train_loader,test_loader,x_train,x_test,adj_train,adj_test,optimizer,device,model)
    
    return ret 



process_type='adj'
list0=[[50,1],[50,50,1],[1]]
c_idx=[]
t_consume=[]
for dataid in range(3,13):
    for j in range(3):
        
        for i in range(10):
            torch.manual_seed(i)
            t0=time.time()
            if process_type=='adj':
                cidx=experiment(data_num=dataid,process_type='adj',epoch=2000,n_channels=4,jump_type='max',gnn_type=1,gnn_k=2\
                           ,survpath='C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer' \
                           if os.name=='nt' else '/home/yilun/remote_dir',node_list=list0[j])
            else:
                cidx=experiment(data_num=dataid,process_type='edge',epoch=1000,n_channels=20,jump_type='max',gnn_type=12,gnn_k=2\
                       ,survpath='C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer' \
                       if os.name=='nt' else '/home/yilun/remote_dir')
            t1=time.time()-t0
            c_idx.append(cidx)
            t_consume.append(t1)
            #'C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer'
        pd.DataFrame({'c_idx':c_idx,'timing':t_consume}).to_csv('C:/Users/Administrator/Desktop/deepgraphsurv/result_dir/simu_result_{}_{}.csv' \
                    if os.name=='nt' else '/home/yilun/result_dir/simu_result_{}.csv'.format(i,j))
