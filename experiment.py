# -*- coding: utf-8 -*-
"""
Created on Mon May  6 20:27:01 2019

@author: Administrator
"""
import torch
import torch.nn as nn

import os.path as osp
from math import ceil
import pandas as pd
import torch
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

class GNN(torch.nn.Module):
    def __init__(self,
                 in_channels=1,
                 hidden_channels=1,
                 out_channels=1,
                 normalize=False,
                 add_loop=False,
                 gnn_k=1,
                 gnn_type=1):
        super(GNN, self).__init__()

        self.add_loop = add_loop

        
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.k=gnn_k#number of repitiions of gnn
        self.gnn_type=gnn_type
        if gnn_type==0:
            self.conv1 = DenseSAGEConv(in_channels=1, out_channels=hidden_channels, normalize=False)
            self.conv2 = DenseSAGEConv(in_channels=hidden_channels, out_channels=out_channels, normalize=False)
        if gnn_type==1:
            self.conv1 = DenseSAGEConv(in_channels=1, out_channels=hidden_channels, normalize=True)
            self.conv2 = DenseSAGEConv(in_channels=hidden_channels, out_channels=out_channels, normalize=True)
        
        if gnn_type==2:
            self.conv1 = GCNConv(in_channels=1, out_channels=hidden_channels, cached=True)
            self.conv2 = GCNConv(in_channels=hidden_channels, out_channels=out_channels, cached=True)
        if gnn_type==3:
            self.conv1 = GCNConv(in_channels=1, out_channels=hidden_channels,improved=True, cached=True)
            self.conv2 = GCNConv(in_channels=hidden_channels, out_channels=out_channels,improved=True, cached=True)
        if gnn_type==4:
            self.conv1 = ChebConv(in_channels=1, out_channels=hidden_channels,K=2)
            self.conv2 = ChebConv(in_channels=hidden_channels, out_channels=out_channels,K=2)
        if gnn_type==5:
            self.conv1 = ChebConv(in_channels=1, out_channels=hidden_channels,K=4)
            self.conv2 = ChebConv(in_channels=hidden_channels, out_channels=out_channels,K=4)
        if gnn_type==6:
            self.conv1 = GraphConv(in_channels=1, out_channels=hidden_channels,aggr='add')
            self.conv2 = GraphConv(in_channels=hidden_channels, out_channels=out_channels,aggr='add')
        if gnn_type==7:
            self.conv1 = GatedGraphConv(in_channels=1,out_channels=hidden_channels, num_layers=3, aggr='add', bias=True)
            self.conv2 = GatedGraphConv(in_channels=hidden_channels,out_channels=out_channels, num_layers=3, aggr='add', bias=True)
        if gnn_type==8:
            self.conv1 = GatedGraphConv(in_channels=1,out_channels=hidden_channels, num_layers=7, aggr='add', bias=True)
            self.conv2 = GatedGraphConv(in_channels=hidden_channels,out_channels=out_channels, num_layers=7, aggr='add', bias=True)
        if gnn_type==9:
            self.conv1 =GATConv(in_channels=1,out_channels=hidden_channels, heads=1, concat=True, negative_slope=0.2,dropout=0.6)
            self.conv2 =GATConv(in_channels=hidden_channels,out_channels=out_channels, heads=1, concat=True, negative_slope=0.2,dropout=0.6)
        if gnn_type==10:
            self.conv1 =GATConv(in_channels=1,out_channels=hidden_channels, heads=6, concat=False, negative_slope=0.2,dropout=0.6)
            self.conv2 =GATConv(in_channels=hidden_channels,out_channels=out_channels, heads=6, concat=False, negative_slope=0.2,dropout=0.6)
            
        if gnn_type==11:
            self.conv1 =GATConv(in_channels=1,out_channels=hidden_channels, heads=4, concat=True, negative_slope=0.2,dropout=0.6)
            self.conv2 =GATConv(in_channels=hidden_channels,out_channels=out_channels, heads=4, concat=True, negative_slope=0.2,dropout=0.6)
        
        if gnn_type==12:
            self.conv1 =GATConv(in_channels=1,out_channels=hidden_channels, heads=4, concat=False, negative_slope=0.2,dropout=0.6)
            self.conv2 =GATConv(in_channels=hidden_channels,out_channels=out_channels, heads=4, concat=False, negative_slope=0.2,dropout=0.6)
            
        if gnn_type==13:
            self.conv1 = AGNNConv(requires_grad=True)
            self.conv2 = AGNNConv(requires_grad=True)
        if gnn_type==14:
            self.conv1 = ARMAConv(in_channels=1, out_channel=hidden_channels, num_stacks=1, num_layers=1, \
                                  shared_weights=False, act=F.relu, dropout=0.5, bias=True)
            self.conv2 = ARMAConv(in_channels=hidden_channels, out_channel=out_channels, num_stacks=1, num_layers=1, \
                                  shared_weights=False, act=F.relu, dropout=0.5, bias=True)
        if gnn_type==15:
            self.conv1 = SGConv(in_channels=1, out_channels=hidden_channels, K=1, cached=True, bias=True)
            self.conv2 = SGConv(in_channels=hidden_channels, out_channels=out_channels, K=1, cached=True, bias=True)
        if gnn_type==16:
            self.conv1 = SGConv(in_channels=1, out_channels=hidden_channels, K=3, cached=True, bias=True)
            self.conv2 = SGConv(in_channels=hidden_channels, out_channels=out_channels, K=3, cached=True, bias=True)
        if gnn_type==17:
            self.conv1 = APPNP(K=1, alpha=0.2, bias=True)
            self.conv2 = APPNP(K=1, alpha=0.2, bias=True)
        if gnn_type==18:
            self.conv1 = APPNP(K=3, alpha=0.2, bias=True)
            self.conv2 = APPNP(K=3, alpha=0.2, bias=True)
        if gnn_type==19:
            self.conv1 =RGCNConv(in_channels=1, out_channels=hidden_channels, num_relations=3, num_bases=2, bias=True)
            self.conv2 =RGCNConv(in_channels=hidden_channels, out_channels=out_channels, num_relations=3, num_bases=2, bias=True)
# =============================================================================
#         if gnn_type==20:
#             self.conv1 = SignedConv(in_channels=1, out_channels=hidden_channels, first_aggr=True, bias=True)
#             self.conv2 = SignedConv(in_channels=hidden_channels, out_channels=out_channels, first_aggr=True, bias=True)
#         if gnn_type==21:
#             self.conv1 =SignedConv(in_channels=1, out_channels=hidden_channels, first_aggr=False, bias=True)
#             self.conv2 =SignedConv(in_channels=hidden_channels, out_channels=out_channels, first_aggr=False, bias=True)
#         if gnn_type==22:
#             self.conv1 = GMMConv(in_channels=1, out_channels=hidden_channels, dim=2, kernel_size=3, bias=True)
#             self.conv2 = GMMConv(in_channels=hidden_channels, out_channels=out_channels, dim=2, kernel_size=3, bias=True)
#         if gnn_type==23:
#             self.conv1 = GMMConv(in_channels=1, out_channels=hidden_channels, dim=5, kernel_size=3, bias=True)
#             self.conv2 = GMMConv(in_channels=hidden_channels, out_channels=out_channels, dim=5, kernel_size=3, bias=True)
#         if gnn_type==24:
#             self.conv1 = GMMConv(in_channels=1, out_channels=hidden_channels, dim=2, kernel_size=3, bias=True)
#             self.conv2 = GMMConv(in_channels=hidden_channels, out_channels=out_channels, dim=2, kernel_size=3, bias=True)
# =============================================================================
        if gnn_type==25:
            self.conv1 = SplineConv(in_channels=1, out_channels=hidden_channels, dim=2, kernel_size=3, is_open_spline=True, \
                                    degree=1, norm=True, root_weight=True, bias=True)
            self.conv2 = SplineConv(in_channels=hidden_channels, out_channels=out_channels, dim=2, kernel_size=3, is_open_spline=True, \
                                    degree=1, norm=True, root_weight=True, bias=True)
        if gnn_type==26:
            self.conv1 = SplineConv(in_channels=1, out_channels=hidden_channels, dim=3, kernel_size=3, is_open_spline=False, \
                                    degree=1, norm=True, root_weight=True, bias=True)
            self.conv2 = SplineConv(in_channels=hidden_channels, out_channels=out_channels, dim=3, kernel_size=3, is_open_spline=False, \
                                    degree=1, norm=True, root_weight=True, bias=True)
        if gnn_type==27:
            self.conv1 = SplineConv(in_channels=1, out_channels=hidden_channels, dim=3, kernel_size=6, is_open_spline=True, \
                                    degree=1, norm=True, root_weight=True, bias=True)
            self.conv2 = SplineConv(in_channels=hidden_channels, out_channels=out_channels, dim=3, kernel_size=6, is_open_spline=True, \
                                    degree=1, norm=True, root_weight=True, bias=True)
        if gnn_type==28:
            self.conv1 = SplineConv(in_channels=1, out_channels=hidden_channels, dim=3, kernel_size=3, is_open_spline=True, \
                                    degree=3, norm=True, root_weight=True, bias=True)
            self.conv2 = SplineConv(in_channels=hidden_channels, out_channels=out_channels, dim=3, kernel_size=3, is_open_spline=True, \
                                    degree=3, norm=True, root_weight=True, bias=True)
        if gnn_type==29:
            self.conv1 = SplineConv(in_channels=1, out_channels=hidden_channels, dim=3, kernel_size=6, is_open_spline=True, \
                                    degree=3, norm=True, root_weight=True, bias=True)
            self.conv2 = SplineConv(in_channels=hidden_channels, out_channels=out_channels, dim=3, kernel_size=6, is_open_spline=True, \
                                    degree=3, norm=True, root_weight=True, bias=True)

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()
        if self.gnn_type<=1:
            if self.k >1 :
                
                x = F.relu(self.bn1(self.conv1(x, adj, mask, self.add_loop)))
                for i in range(self.k-1):
                    x  = F.relu(self.bn2(self.conv2(x, adj, mask, self.add_loop)))
            else:
                x = F.relu(self.bn1(self.conv1(x, adj, mask, self.add_loop)))
        
            x = self.bn(1, F.relu(self.conv1(x, adj, mask, self.add_loop)))
        elif self.gnn_type in [2,3,4,5,14,15,16,17,18]:
            if self.k >1 :
                
                x = F.relu(self.bn1( self.conv1(x, edge_index)))
                for i in range(self.k-1):
                    x = F.relu(self.bn2( self.conv2(x, edge_index)))
            else:
                x = F.relu(self.bn1( self.conv1(x, edge_index)))
        elif self.gnn_type!=19:
            if self.k >1 :
                
                x = F.relu(self.bn1( self.conv1(x, edge_index,edge_weight)))
                for i in range(self.k-1):
                    x = F.relu(self.bn2( self.conv2(x, edge_index,edge_weight)))
            else:
                x = F.relu(self.bn1( self.conv1(x, edge_index,edge_weight)))
        

        return x

class Net0(torch.nn.Module):
    def __init__(self):
        super(Net0, self).__init__()

        
        self.gnn1= GNN(1, 30, 6, gnn_type=1,gnn_k=1,add_loop=True)
        self.gnn2 = GNN(100, 64, 64, gnn_type=1,add_loop=True)

        self.bn1 = torch.nn.BatchNorm1d(64)

        self.lin1 = torch.nn.Linear(30, 30)
        self.lin2 = torch.nn.Linear(64, 6)

    def forward(self, x, adj, mask=None):
        x = self.gnn1(x, adj, mask)
        #x = self.gnn2(x, adj, mask)

        x= self.bn1()

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        #x = self.lin2(x)
        return x


class Net_el(torch.nn.Module):#GAT gnn_type=9-12,ARGNN,gnn_type=13
    def __init__(self):
        super(Net_el, self).__init__()

        
        self.gnn1= GNN(1, 30, 6, gnn_type=10,gnn_k=1)
        self.gnn2 = GNN(100, 64, 64, gnn_type=10,gnn_k=1)

        self.bn1 = torch.nn.BatchNorm1d(64)

        self.lin1 = torch.nn.Linear(30, 1)
        self.lin2 = torch.nn.Linear(64, 6)

    def forward(self, x, el ):
        x = self.gnn1(x, el)
        #x = self.gnn2(x, adj, mask)

        x= self.bn1(x)

        #x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        #x = self.lin2(x)
        return x

















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
	'''Calculate the average Cox negative partial log-likelihood.
	Input:
		pred: linear predictors from trained model.
		ytime: true survival time from load_data().
		yevent: true censoring status from load_data().
	Output:
		cost: the cost that is to be minimized.
	'''
	n_observed = yevent.sum(0)
	ytime_indicator = R_set(ytime)
	###if gpu is being used
	if torch.cuda.is_available():
		ytime_indicator = ytime_indicator.cuda()
	###
	risk_set_sum = ytime_indicator.mm(torch.exp(pred)) 
	diff = pred - torch.log(risk_set_sum)
	sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
	cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))

	return(cost)


def c_index(pred, ytime, yevent):
	'''Calculate concordance index to evaluate models.
	Input:
		pred: linear predictors from trained model.
		ytime: true survival time from load_data().
		yevent: true censoring status from load_data().
	Output:
		concordance_index: c-index (between 0 and 1).
	'''
	n_sample = len(ytime)
	ytime_indicator = R_set(ytime)
	ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
	###T_i is uncensored
	censor_idx = (yevent == 0).nonzero()
	zeros = torch.zeros(n_sample)
	ytime_matrix[censor_idx, :] = zeros
	###1 if pred_i < pred_j; 0.5 if pred_i = pred_j
	pred_matrix = torch.zeros_like(ytime_matrix)
	for j in range(n_sample):
		for i in range(n_sample):
			if pred[i] < pred[j]:
				pred_matrix[j, i]  = 1
			elif pred[i] == pred[j]: 
				pred_matrix[j, i] = 0.5
	
	concord_matrix = pred_matrix.mul(ytime_matrix)
	###numerator
	concord = torch.sum(concord_matrix)
	###denominator
	epsilon = torch.sum(ytime_matrix)
	###c-index = numerator/denominator
	concordance_index = torch.div(concord, epsilon)
	###if gpu is being used
	if torch.cuda.is_available():
		concordance_index = concordance_index.cuda()
	###
	return(concordance_index)



def get_data_list_el(x_matrix,el):  
    data_list=[]
      
    for i in range(1,x_matrix.shape[1]):
        data_list+=[Data(x=torch.tensor(x_matrix.values[:,i].reshape((-1,1))),edge_list=el)]
    return data_list



def get_data_list_el_w(x_matrix,el,eattr):  
    data_list=[]
    for i in range(1,x_matrix.shape[1]):
        data_list+=[Data(x=torch.tensor(x_matrix.values[:,i].reshape((-1,1))),\
                         edge_list=el,edge_attr=eattr.reshape((-1,1)))]
    return data_list




def get_data_adj(x_matrix,adj):  
    data_list=[]
      
    for i in range(x_matrix.shape[1]):
        data_list+=[Data(x=torch.tensor(x_matrix.values[:,i].reshape((-1,1))),adj=torch.tensor(np.array(adj).astype(int32)))]
    return data_list


def train(epoch, y_train, censor_train):
    model.train()
    

    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = neg_par_log_likelihood(output, y_train, censor_train)#event=0,censored
        loss.backward()
        
        optimizer.step()
    


def test(loader,y_test,censor_test):
    model.eval()
    

     
    for data in test_loader:
        data = data.to(device)
        pred = model(data.x,  data.edge_index)
        cidx = c_index(pred, y_test, censor_test)
    return cidx

def do(maxit, y_train, censor_train, y_test, censor_test):
    for epoch in range(1, maxit):
        train(epoch, y_train, censor_train)
        cidx = test(test_loader, y_test, censor_test)
        if epoch %10==2:
            print('c_index = {}'.format(cidx))




def get_data_edge_lung(data_num,data_type='edge'):
    train_rate=0.6
    
    data_num=0
    x=pd.read_csv('C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/survdata/lung_x_{}.csv'.format(data_num))
    y=pd.read_csv('C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/survdata/lung_t_{}.csv'.format(data_num)).iloc[:,1].values
    censor=pd.read_csv('C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/survdata/lung_c_{}.csv'.format(data_num)).iloc[:,1].values
    edl=pd.read_csv('C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/survdata/dt_{}.csv'.format(data_num)).iloc[:,1:]
    
    n=x.shape[1]-1
    
    x=x.take(np.argsort(y),axis=1)
    y=y[np.argsort(y)]
    censor=censor[np.argsort(y)]


    if data_type=='edge':
        ev=edl.values
        edges=[]
        for i in range(ev.shape[0]):
            edges+=[[int(ev[i,0]),int(ev[i,1])]]
        edges_tc=torch.from_numpy(np.array(edges)).long().t().contiguous()
        data_list=get_data_list_el(x,edges_tc)
        #loader = DataLoader(data_list, batch_size=len(data_list),shuffle=False)
    elif data_type=='edge_weight':
    
    
        edges_tc=torch.from_numpy(np.array(edges)).long().t().contiguous()
        edges_weight=[]
        for i in range(ev.shape[0]):
            edges_weight+=[(int(ev[i,0]),int(ev[i,1]),ev[i,2])]
        data_list=get_data_list_el_w(x,edges_tc,ev[:,2])
        #loader = DataLoader(data_list, batch_size=len(data_list),shuffle=False)
    else:
        G=nx.Graph()
        #G.add_nodes_from([1,2,3])
        #G.add_weighted_edges_from([(1,2,0.5),(1,4,0.5)])
        G.add_nodes_from(x.iloc[:,0])
        G.add_weighted_edges_from(edges_weight)
        
        adj_bi=nx.adjacency_matrix(G,weight=None).todense()
        adj_weight=nx.adjacency_matrix(G).todense()
        data_list=get_data_adj(x,adj_bi)
        adj_train=np.tile(np.array(adj_bi),(int(train_rate*n),1,1))
        adj_test=np.tile(np.array(adj_bi),(n-int(train_rate*n),1,1))
        #loader=DenseDataLoader(data_list, batch_size=len(data_list),shuffle=False)   
    trainset=np.sort(np.random.choice(n,size=int(train_rate*n),replace=False))
    testset=np.sort(np.setdiff1d(range(n),trainset))

    return data_list,trainset,testset
#create dataset
#with edge list
#with weight
 
data_list,trainset,testset=get_data_edge_lung(data_num,data_type='edge')


    


train_list=[data_list[i] for i in trainset]
test_list=[data_list[i] for i in testset]
y_train=np.array([y[i] for i in trainset])
y_test=np.array([y[i] for i in trainset])
censor_train=np.array([y[i] for i in trainset])
censor_test=np.array([y[i] for i in trainset])



#Global
train_loader=DataLoader(train_list,batch_size=len(train_list),shuffle=False)
test_loader=DataLoader(test_list,batch_size=len(test_list),shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net_el().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
   
do(150)

