# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:29:04 2019

@author: Administrator
"""

import torch
import torch.nn as nn



import pandas as pd

import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader,DataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool,GCNConv,ChebConv
from torch_geometric.nn import GraphConv,GatedGraphConv,GATConv,AGNNConv,ARMAConv,SGConv,APPNP,RGCNConv
from torch_geometric.nn import SignedConv,GMMConv
from torch_geometric.nn import SplineConv
from torch_geometric.nn import global_sort_pool,GlobalAttention,Set2Set
from torch_geometric.nn import TopKPooling,SAGEConv

from torch import randperm
from torch_geometric.nn import max_pool,avg_pool,max_pool_x,avg_pool_x,graclus
from torch_geometric.nn import JumpingKnowledge,DeepGraphInfomax
from torch_geometric.nn import InnerProductDecoder,GAE,VGAE,ARGA,ARGVA

from torch_geometric.data import Data, DataLoader
import networkx as nx
import numpy as np

from torch_geometric.utils import from_networkx

from torch_geometric.utils import is_undirected, to_undirected












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


def neg_par_log_likelihood(pred, ytime, yevent):#event=0,censored
    #ytime should be sorted with increasing order
    #yevent=torch.tensor(yevent)#
    #yevent=yevent.view(-1,1).type(torch.float)  
    #pred=pred.view(-1)
    n_observed = int(yevent.sum(0))  
    ytime_indicator = R_set(ytime)
	###if gpu is being used
    if torch.cuda.is_available():
	    ytime_indicator = ytime_indicator.cuda()
	###
    risk_set_sum = ytime_indicator.mm(torch.exp(pred)) 
    diff = pred - torch.log(risk_set_sum)
    yevent=yevent.view(-1,1).type(torch.float)  
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
    cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))
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
    denomi=0.
    nume=0.
    for j in uncensor_idx[:-1]:
        for i in range(j+1,n_sample):
            if ytime[i] > ytime[j]:
                denomi = denomi+1
                if pred[i] > pred[j]:
                    nume=nume+1
            
    
	###c-index = numerator/denominator
    concordance_index = torch.div(nume, denomi)
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




def get_data_adj(x_matrix,adj):  
    
    x_matrix=x_matrix[:,1:] 
    x=torch.from_numpy(x_matrix).type(torch.float32).t().contiguous().view(x_matrix.shape[1],x_matrix.shape[0],1)
    return x







def get_data_edge_lung(data_num,data_type='edge',train_rate=0.6):
    
    
    
    x=pd.read_csv('C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/survdata/lung_x_{}.csv'.format(data_num))
    y=pd.read_csv('C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/survdata/lung_t_{}.csv'.format(data_num)).iloc[:,1].values
    censor=pd.read_csv('C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/survdata/lung_c_{}.csv'.format(data_num)).iloc[:,1].values
    edl=pd.read_csv('C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/survdata/dt_{}.csv'.format(data_num)).iloc[:,1:]
    #G=nx.Graph()
    n=x.shape[1]-1
    #each column, except for the column 0 of x ,represents features for a sample
    #the features are in alphabetical order,this should agree with the order
    #in edl
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
        for i in G.nodes:
            edges_weight+=[(i,i,1.)]
        
        G.add_weighted_edges_from(edges_weight)
        G = nx.DiGraph(G) 
        
        adj_bi=nx.adjacency_matrix(G,weight=None).todense()
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
        adj_weight=nx.adjacency_matrix(G).todense()
        G1=nx.from_numpy_matrix(adj_weight)
        data_list=get_data_adj(x.values,adj_bi)
        
        adjt=torch.tensor(adj_weight.astype(np.float32))
        adj_train=adjt.repeat((int(train_rate*n),1)).view(-1,adjt.size(0),adjt.size(0))
        adj_test=adjt.repeat((n-int(train_rate*n),1)).view(-1,adjt.size(0),adjt.size(0))
        #loader=DenseDataLoader(data_list, batch_size=len(data_list),shuffle=False)   
    trainset=np.sort(np.random.choice(n,size=int(train_rate*n),replace=False))
    testset=np.sort(np.setdiff1d(range(n),trainset))
    if data_type=='edge' or data_type=='edge_weight':
        return data_list,y,censor,trainset,testset,x.shape[0],None,None
    else:
        return data_list,y,censor,trainset,testset,x.shape[0],adj_train,adj_test
#create dataset
#with edge list
#with weight
 



class GNN(torch.nn.Module):
    def __init__(self,
                 in_channels=1,
                 hidden_channels=1,
                 out_channels=1,
                 normalize=False,
                 add_loop=False,
                 gnn_k=1,
                 gnn_type=1,
                 jump=None,#None,max,lstm
                 res=False,
                 activation='leaky'
                 ):
        super(GNN, self).__init__()

        self.add_loop = add_loop

        self.in_channels=in_channels
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.k=gnn_k#number of repitiions of gnn
        self.gnn_type=gnn_type
        
        self.jump=jump
        if not( jump is None):
            if jump!='lstm':
                self.jk=JumpingKnowledge(jump)
            else:
                self.jk=JumpingKnowledge(jump,out_channels,gnn_k)
        if activation=='leaky':
            self.activ=F.leaky_relu
        elif activation=='elu':
            self.activ=F.elu
        elif activation=='relu':
            self.activ=F.relu
        self.res=res
        if self.gnn_type in [10,12] and self.res==True:
            raise Exception('res must be false when gnn_type==10 or 12!')
        if self.k ==1 and self.res==True:
            raise Exception('res must be false when gnn_k==1!')
        if self.k ==1 and not( self.jump is  None):
            raise Exception('jumping knowledge only serves for the case where k>1!')
        if gnn_type==0:
            self.conv1 = DenseSAGEConv(in_channels=self.in_channels, out_channels=out_channels, normalize=False)
            self.conv2 = DenseSAGEConv(in_channels=hidden_channels, out_channels=out_channels, normalize=False)
        if gnn_type==1:
            self.conv1 = DenseSAGEConv(in_channels=self.in_channels, out_channels=out_channels, normalize=True)
            self.conv2 = DenseSAGEConv(in_channels=hidden_channels, out_channels=out_channels, normalize=True)
        
        if gnn_type==2:
            self.conv1 = GCNConv(in_channels=1, out_channels=out_channels, cached=False)
            self.conv2 = GCNConv(in_channels=hidden_channels, out_channels=out_channels, cached=False)
        if gnn_type==3:
            self.conv1 = GCNConv(in_channels=1, out_channels=out_channels,improved=True, cached=False)
            self.conv2 = GCNConv(in_channels=hidden_channels, out_channels=out_channels,improved=True, cached=False)
        if gnn_type==4:
            self.conv1 = ChebConv(in_channels=1, out_channels=out_channels,K=2)
            self.conv2 = ChebConv(in_channels=hidden_channels, out_channels=out_channels,K=2)
        if gnn_type==5:
            self.conv1 = ChebConv(in_channels=1, out_channels=out_channels,K=4)
            self.conv2 = ChebConv(in_channels=hidden_channels, out_channels=out_channels,K=4)
        if gnn_type==6:
            self.conv1 = GraphConv(in_channels=1, out_channels=out_channels,aggr='add')
            self.conv2 = GraphConv(in_channels=hidden_channels, out_channels=out_channels,aggr='add')
        if gnn_type==7:
            self.conv1 = GatedGraphConv(out_channels=out_channels, num_layers=3, aggr='add', bias=True)
            self.conv2 = GatedGraphConv(out_channels=out_channels, num_layers=3, aggr='add', bias=True)
        if gnn_type==8:
            self.conv1 = GatedGraphConv(out_channels=out_channels, num_layers=7, aggr='add', bias=True)
            self.conv2 = GatedGraphConv(out_channels=out_channels, num_layers=7, aggr='add', bias=True)
        if gnn_type==9:
            self.conv1 =GATConv(in_channels=1,out_channels=out_channels, heads=1, concat=True, negative_slope=0.2,dropout=0)
            self.conv2 =GATConv(in_channels=hidden_channels,out_channels=out_channels, heads=1, concat=True, negative_slope=0.2,dropout=0.6)
        if gnn_type==10:
            self.conv1 =GATConv(in_channels=1,out_channels=out_channels, heads=6, concat=False, negative_slope=0.2,dropout=0.6)
            self.conv2 =GATConv(in_channels=hidden_channels,out_channels=out_channels, heads=6, concat=False, negative_slope=0.2,dropout=0.6)
            
        if gnn_type==11:
            self.conv1 =GATConv(in_channels=1,out_channels=out_channels, heads=4, concat=True, negative_slope=0.2,dropout=0)
            self.conv2 =GATConv(in_channels=hidden_channels,out_channels=out_channels, heads=4, concat=True, negative_slope=0.2,dropout=0.6)
        
        if gnn_type==12:
            self.conv1 =GATConv(in_channels=1,out_channels=out_channels, heads=4, concat=False, negative_slope=0.2,dropout=0.6)
            self.conv2 =GATConv(in_channels=hidden_channels,out_channels=out_channels, heads=4, concat=False, negative_slope=0.2,dropout=0.6)
            
        if gnn_type==13:
            self.conv1 = AGNNConv(requires_grad=True)
            self.conv2 = AGNNConv(requires_grad=True)
        if gnn_type==14:
            self.conv1 = ARMAConv(in_channels=1, out_channels=hidden_channels, num_stacks=1, num_layers=1, 
                                  shared_weights=False, act=F.relu, dropout=0.5, bias=True)
            self.conv2 = ARMAConv(in_channels=hidden_channels, out_channels=out_channels, num_stacks=1, num_layers=1, 
                                  shared_weights=False, act=F.relu, dropout=0.5, bias=True)
        if gnn_type==15:
            self.conv1 = SGConv(in_channels=1, out_channels=out_channels, K=1, cached=True, bias=True)
            self.conv2 = SGConv(in_channels=hidden_channels, out_channels=out_channels, K=1, cached=True, bias=True)
        if gnn_type==16:
            self.conv1 = SGConv(in_channels=1, out_channels=out_channels, K=3, cached=True, bias=True)
            self.conv2 = SGConv(in_channels=hidden_channels, out_channels=out_channels, K=3, cached=True, bias=True)
        if gnn_type==17:
            self.conv1 = APPNP(K=1, alpha=0.2, bias=True)
            self.conv2 = APPNP(K=1, alpha=0.2, bias=True)
        if gnn_type==18:
            self.conv1 = APPNP(K=3, alpha=0.2, bias=True)
            self.conv2 = APPNP(K=3, alpha=0.2, bias=True)
        if gnn_type==19:
            self.conv1 =RGCNConv(in_channels=1, out_channels=out_channels, num_relations=3, num_bases=2, bias=True)
            self.conv2 =RGCNConv(in_channels=hidden_channels, out_channels=out_channels, num_relations=3, num_bases=2, bias=True)
# =============================================================================
#         if gnn_type==20:
#             self.conv1 = SignedConv(in_channels=1, out_channels=out_channels, first_aggr=True, bias=True)
#             self.conv2 = SignedConv(in_channels=hidden_channels, out_channels=out_channels, first_aggr=True, bias=True)
#         if gnn_type==21:
#             self.conv1 =SignedConv(in_channels=1, out_channels=out_channels, first_aggr=False, bias=True)
#             self.conv2 =SignedConv(in_channels=hidden_channels, out_channels=out_channels, first_aggr=False, bias=True)
#         if gnn_type==22:
#             self.conv1 = GMMConv(in_channels=1, out_channels=out_channels, dim=2, kernel_size=3, bias=True)
#             self.conv2 = GMMConv(in_channels=hidden_channels, out_channels=out_channels, dim=2, kernel_size=3, bias=True)
#         if gnn_type==23:
#             self.conv1 = GMMConv(in_channels=1, out_channels=out_channels, dim=5, kernel_size=3, bias=True)
#             self.conv2 = GMMConv(in_channels=hidden_channels, out_channels=out_channels, dim=5, kernel_size=3, bias=True)
#         if gnn_type==24:
#             self.conv1 = GMMConv(in_channels=1, out_channels=out_channels, dim=2, kernel_size=3, bias=True)
#             self.conv2 = GMMConv(in_channels=hidden_channels, out_channels=out_channels, dim=2, kernel_size=3, bias=True)
# =============================================================================
        if gnn_type==25:
            self.conv1 = SplineConv(in_channels=1, out_channels=out_channels, dim=2, kernel_size=3, is_open_spline=True, 
                                    degree=1, norm=True, root_weight=True, bias=True)
            self.conv2 = SplineConv(in_channels=hidden_channels, out_channels=out_channels, dim=2, kernel_size=3, is_open_spline=True, 
                                    degree=1, norm=True, root_weight=True, bias=True)
        if gnn_type==26:
            self.conv1 = SplineConv(in_channels=1, out_channels=out_channels, dim=3, kernel_size=3, is_open_spline=False, 
                                    degree=1, norm=True, root_weight=True, bias=True)
            self.conv2 = SplineConv(in_channels=hidden_channels, out_channels=out_channels, dim=3, kernel_size=3, is_open_spline=False, 
                                    degree=1, norm=True, root_weight=True, bias=True)
        if gnn_type==27:
            self.conv1 = SplineConv(in_channels=1, out_channels=out_channels, dim=3, kernel_size=6, is_open_spline=True, 
                                    degree=1, norm=True, root_weight=True, bias=True)
            self.conv2 = SplineConv(in_channels=hidden_channels, out_channels=out_channels, dim=3, kernel_size=6, is_open_spline=True, 
                                    degree=1, norm=True, root_weight=True, bias=True)
        if gnn_type==28:
            self.conv1 = SplineConv(in_channels=1, out_channels=out_channels, dim=3, kernel_size=3, is_open_spline=True, 
                                    degree=3, norm=True, root_weight=True, bias=True)
            self.conv2 = SplineConv(in_channels=hidden_channels, out_channels=out_channels, dim=3, kernel_size=3, is_open_spline=True, 
                                    degree=3, norm=True, root_weight=True, bias=True)
        if gnn_type==29:
            self.conv1 = SplineConv(in_channels=1, out_channels=out_channels, dim=3, kernel_size=6, is_open_spline=True, 
                                    degree=3, norm=True, root_weight=True, bias=True)
            self.conv2 = SplineConv(in_channels=hidden_channels, out_channels=out_channels, dim=3, kernel_size=6, is_open_spline=True, 
                                    degree=3, norm=True, root_weight=True, bias=True)

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
        if self.gnn_type<=1:
            #batch_size, num_nodes, in_channels = x.size()
        #batch_size,num_nodes,num_nodes=adj.size()
            
            batch_size, num_nodes, _ = x.size()
            if self.k >1 :
                
                x = self.conv1(x, adj)
                x = x.view(-1,x.size()[-1])
                x = self.activ(self.bn2(x))
                if self.res==True:
                    x0 = x
# =============================================================================
#                     print('input size')
#                     print(x.size())
# =============================================================================
                if self.jump==None:
                    for i in range(self.k-1):
                        x = x.view(batch_size, num_nodes, -1)
                        x = self.conv2(x, adj)
                        x = x.view(-1,x.size()[-1])
                        x = self.activ(self.bn2(x))
                else :
                    xs=[x]
                    for i in range(self.k-1):
                        x = x.view(batch_size, num_nodes, -1)
                        x = self.conv2(x, adj)
                        x = x.view(-1,x.size()[-1])
                        x = self.activ(self.bn2(x))
                        xs = xs + [x]
                    x = self.jk(xs)
                if self.res==True:
# =============================================================================
#                     print('output size')
#                     print(x.size())
# =============================================================================
                    x = x + x0    
                        
            else:
                x = self.conv1(x, adj)
                x = x.view(-1,x.size()[-1])
                x = self.activ(self.bn2(x))
            x = x.view(batch_size, num_nodes, -1)
            
        #GAT,AGNN,GraphConv, only accepts edge_index   [6,9,10,11,12]  
        elif self.gnn_type in [2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18] and edge_weight is None:
            
            
            if self.k >1 :
                
                x = self.activ(self.bn1( self.conv1(x, edge_index)))
                if self.res==True:
                    x0 = x
# =============================================================================
#                     print('input size')
#                     print(x.size())
# =============================================================================
                if self.jump==None:
                    for i in range(self.k-1):
                        x = self.activ(self.bn2( self.conv2(x, edge_index)))
                else:
                    xs=[x]
                    for i in range(self.k-1):
                        x = self.activ(self.bn2( self.conv2(x, edge_index)))
                        xs = xs + [x]
                    x = self.jk(xs)
                if self.res==True:
# =============================================================================
#                     print('output size')
#                     print(x.size())
# =============================================================================
                    x = x + x0
            else:
                x = self.activ(self.bn1( self.conv1(x, edge_index)))
            
        elif self.gnn_type in [2,3,4,5,7,8,14,15,16,17,18] and not (edge_weight is None):
            
            if self.res==True:
                x0 = x
                
            if self.k >1 :
                
                x = self.activ(self.bn1( self.conv1(x, edge_index,edge_weight)))
                if self.res==True:
                    x0 = x
# =============================================================================
                    #print('input size')
                    #print(x.size())
# =============================================================================
                if self.jump==None:
                    for i in range(self.k-1):
                        x = self.activ(self.bn2( self.conv2(x, edge_index,edge_weight)))
                else:
                    xs=[x]
                    for i in range(self.k-1):
                        x = self.activ(self.bn2( self.conv2(x, edge_index,edge_weight)))
                        xs = xs + [x]
                    x = self.jk(xs)
                if self.res==True:
# =============================================================================
                    #print('output size')
                    #print(x.size())
# =============================================================================
                    x = x + x0
            else:
                x = self.activ(self.bn1( self.conv1(x, edge_index,edge_weight)))
            if self.res==True:
                x = x + x0
        else:
            raise ValueError()

        return x


class Net0(torch.nn.Module):#gcn
    def __init__(self,num_features,hidden_channels=5,out_channels=5,gnn_k=1,gnn_type=2,activation='leaky',res=False,jump=None):
        super(Net0, self).__init__()
        
        self.num_features = num_features    
        self.hidden_channels=hidden_channels
        self.out_channels=out_channels
        self.gnn_k=gnn_k
        self.gnn_type=gnn_type
        if activation=='leaky':
            self.activ=F.leaky_relu
        elif activation=='elu':
            self.activ=F.elu
        elif activation=='relu':
            self.activ=F.relu
        self.gnn1= GNN(1, self.hidden_channels, self.out_channels, gnn_type=self.gnn_type,gnn_k=self.gnn_k,add_loop=True,res=res,jump=jump)
        #self.gnn2 = GNN(100, 64, 64, gnn_type=2,add_loop=True)

        #self.bn1 = torch.nn.BatchNorm1d(self.hidden_channels)

        self.lin1 = torch.nn.Linear(self.num_features, 1)
        #self.lin2 = torch.nn.Linear(64, 6)
        self.lin1times1 = torch.nn.Linear(self.out_channels,1)
        
    def forward(self, x, edge_index,edge_weight=None):
        if edge_weight is None:
            x = self.gnn1(x, edge_index)
        else:
            x = self.gnn1(x, edge_index,edge_weight)
        #x = self.gnn2(x, adj, mask)

        #x= self.bn1(x)

        #x = x.mean(dim=1)
        x = x.view(-1,self.num_features,self.out_channels)
        #x = x.mean(dim=-1)
        x = self.activ(self.lin1times1(x))
        x = x.view(x.size()[0:2])
        x = self.activ(self.lin1(x))
        #x = self.lin2(x)
        return x



class Net_diff_pool(torch.nn.Module):
    #note that for gnn2_pool,the input dimension is the number of hidden nodes after the first pooling operation
    def __init__(self,num_features,hidden_channels=50,out_channels=50,out_clusters=200,gnn_k=1,gnn_type=1,activation='leaky',res=False,jump=None):
        super(Net_diff_pool, self).__init__()
        self.num_features = num_features    
        self.out_clusters=out_clusters
        self.out_channels=out_channels
        self.hidden_channels=hidden_channels
        if activation=='leaky':
            self.activ=F.leaky_relu
        elif activation=='elu':
            self.activ=F.elu
        elif activation=='relu':
            self.activ=F.relu
        self.gnn1_pool = GNN(1, out_clusters, out_clusters,gnn_k=gnn_k,gnn_type=1,add_loop=True,res=res,jump=jump)
        self.gnn1_embed = GNN(1, hidden_channels, out_channels,gnn_k=gnn_k,gnn_type=1,add_loop=True,res=res,jump=jump)

        
        self.gnn2_pool = GNN(in_channels=out_channels,hidden_channels=int(out_clusters*0.2),out_channels=int(out_clusters*0.2),gnn_k=gnn_k,gnn_type=1,add_loop=True,res=res,jump=jump)
        self.gnn2_embed = GNN(out_channels, out_channels, out_channels,gnn_k=gnn_k,gnn_type=1,add_loop=True,res=res,jump=jump)

        self.gnn3_embed = GNN(out_channels, out_channels, out_channels,gnn_k=gnn_k,gnn_type=1,add_loop=True,res=res,jump=jump)

        self.lin1 = torch.nn.Linear(int(0.2*self.out_clusters), 1)
        self.lin1times1 = torch.nn.Linear(self.out_channels,1)
        
    

    def forward(self, x_input, adj, mask=None):
        #print('forward diff')
        s = self.gnn1_pool(x_input, adj=adj, mask=mask)
        s = s.view(-1,self.num_features,s.size()[-1])
        x = self.gnn1_embed(x_input, adj=adj, mask=mask)
        x = x.view(-1,self.num_features,x.size()[-1])
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        
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
        s = self.gnn2_pool(x, adj=adj)
        s = s.view(-1,adj.size()[1],s.size()[-1])
        x = self.gnn2_embed(x, adj=adj)
        x = x.view(-1,x.size()[1],x.size()[-1])
        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj=adj)
        x = x.view(-1,int(0.2*self.out_clusters),self.out_channels)
        #x = x.mean(dim=-1)#1*1 convolution
        x = self.activ(self.lin1times1(x))
        x = x.view(x.size()[0:2])
        
        x = self.activ(self.lin1(x))
        
        return x, l1 + l2, e1 + e2





#####################################################end functions

def train_c(train_loader, y_train, censor_train,mode,x_train=None,adj_train=None):
    model.train()
    if mode=='edge'or mode=='edge_weight':
        for data in train_loader:
            data = data.to(device)
            #xdata=data
            optimizer.zero_grad()
            if mode=='edge':
                output = model(data.x.type(torch.float),data.edge_index)
            elif mode=='edge_weight':
                output = model(data.x.type(torch.float), data.edge_index,data.edge_attr.type(torch.float))
            
            c_idx = c_index(output,torch.from_numpy(y_train),torch.from_numpy( censor_train))
            loss = neg_par_log_likelihood(output, torch.from_numpy(y_train), torch.from_numpy(censor_train))#event=0,censored
            loss.backward()
            optimizer.step()
        
        return loss,c_idx
    else:
        x_train = x_train.to(device)
        adj_train=adj_train.to(device)
        #print(x_train is None)
        optimizer.zero_grad()
        output,link_loss,ent_loss = model(x_input=x_train, adj=adj_train.type(torch.float32),mask=None)
         
        
        c_idx = c_index(output,torch.from_numpy(y_train),torch.from_numpy( censor_train))
        loss = neg_par_log_likelihood(output, torch.from_numpy(y_train), torch.from_numpy(censor_train)) + link_loss + ent_loss#event=0,censored
        loss.backward()
        optimizer.step()
        
        return loss,c_idx    
#
#model.gnn1.conv1(data.x.type(torch.float), data.edge_index,data.edge_attr.type(torch.float).view(-1))



def test(test_loader,y_test,censor_test,mode,x_test=None,adj_test=None):
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
            cidx = c_index(pred, torch.from_numpy(y_test), torch.from_numpy(censor_test))
        return cidx
    else:
        x_test = x_test.to(device)
        adj_test=adj_test.to(device)
        #optimizer.zero_grad()
        with torch.no_grad():
            pred ,link_loss,ent_loss= model(x_test, adj=adj_test.type(torch.float32))
            cidx = c_index(pred, torch.from_numpy(y_test), torch.from_numpy(censor_test))
        return cidx

#debug
# =============================================================================
#         model(xdata.x.type(torch.float), xdata.edge_index)
# model.gnn1.conv1(xdata.x.type(torch.float), xdata.edge_index)    
# =============================================================================
#

def do_c(maxit, y_train, censor_train, y_test, censor_test,mode,train_loader,test_loader,x_train,x_test,adj_train,adj_test):
    
    for epoch in range(1, maxit):
        temp_loss,train_cidx=train_c(train_loader, y_train, censor_train,mode,x_train,adj_train)
        test_cidx = test(test_loader, y_test, censor_test,mode,x_test,adj_test)
        if epoch %1==0:
            print('training_loss at epock{} = {},train_c_index = {}'.format(epoch,temp_loss.data.numpy().ravel()[0],train_cidx))
            print('testing_c_index = {}'.format(test_cidx))
    
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
            
data_num=0
process_type='edge_weight' #'edge_list'#'adj'
process_type='edge'
process_type='adj'
if process_type!='adj':
    data_list,y,censor,trainset,testset,num_features,adj_train,adj_test = get_data_edge_lung(data_num,data_type=process_type)
else:
    data_list,y,censor,trainset,testset,num_features,adj_train,adj_test = get_data_edge_lung(data_num,data_type=process_type)

    


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
    x_train=None
    x_test=None
    
else:
    train_loader = None
    test_loader = None
    x_train=data_list[trainset]
    x_test=data_list[testset]
    




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if process_type!='adj':
    #gnn_type=2-3GCONV,9-14GAT,4-5Cheb,7-8GatedGraph,6GraphConv,15SG,14ARMA #6,9,10,11,12 only edge
    model = Net0(num_features,hidden_channels=2,out_channels=2,gnn_type=2,gnn_k=3,res=True,jump='lstm').to(device)
else:
    model = Net_diff_pool(num_features,hidden_channels=2,out_channels=2,out_clusters=20,gnn_k=3,res=True,jump='max').to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

do_c(2, y_train, censor_train, y_test, censor_test,process_type,train_loader,test_loader,x_train,x_test,adj_train,adj_test)

# =============================================================================
# test
#model.eval()

# 
# 
# ################
# model = Net_diff_pool(num_features,hidden_channels=5,out_channels=5,out_clusters=20,k=1).to(device)
# x=x_train
# adj=adj_train.type(torch.float32)  
# x1=model.gnn1_pool.conv1(x, adj )  
# result=model(x_train,adj)
# 
# for data in test_loader:
#     dx=data.x.type(torch.float32)
#     de=data.edge_index
# dx.size()
# de.size()
# 
# =============================================================================
