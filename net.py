# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:43:53 2019

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


from torch_geometric.nn import max_pool,avg_pool,max_pool_x,avg_pool_x,graclus
from torch_geometric.nn import JumpingKnowledge,DeepGraphInfomax
from torch_geometric.nn import InnerProductDecoder,GAE,VGAE,ARGA,ARGVA

from torch_geometric.data import Data, DataLoader
max_nodes = 100

torch.manual_seed(1245)




data_list = [Data(...), ..., Data(...)]
loader = DataLoader(data_list, batch_size=32)

def get_edge_list(adj):
    l=adj.shape[0]
    adj=adj.values[:,1:]
    edgl=[]
    for i in range(l):
        for j in range(l):
            if adj[i,j]!=0:
                edgl+=[[i,j]]
                
    return torch.tensor(edgl).t().contiguous()
        

el=get_edge_list(amatrix)
        
def get_data_list(x_matrix,adj):  
    data_list=[]
    el=get_edge_list(adj)      
    for i in range(x_matrix.shape[0]):
        data_list+=Data(x=x_matrix.values[i,1:],edge_list=el)
    return data_list

def get_data_list_el(x_matrix,el):  
    data_list=[]
      
    for i in range(x_matrix.shape[0]):
        data_list+=Data(x=x_matrix.values[i,1:],edge_list=el)
    return data_list
dl=get_data_list(x_matrix,amatrix)


edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

el=pd.read_csv('C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/dt2.csv')

el=el.iloc[:,1:]

data = Data(x=x, edge_index=edge_index.t().contiguous())

#x_0 (3559)  x_1 (2471) x_2 (1516)  x_3(765) correspond to

data_num=0
x=pd.read_csv('C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/survdata/lung_x_{}.csv'.format(data_num))
y=pd.read_csv('C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/survdata/lung_t_{}.csv'.format(data_num)).iloc[:,1].values
censor=pd.read_csv('C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/survdata/lung_c_{}.csv'.format(data_num)).iloc[:,1].values
edl=pd.read_csv('C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/survdata/dt_{}.csv'.format(data_num)).iloc[:,1:]
G=nx.Graph()
G.clear()
G.add_nodes_from(x.iloc[:,0])

ev=edl.values
edges=[]
for i in range(ev.shape[0]):
    edges+=[[int(ev[i,0]),int(ev[i,1])]]
    
edges_tc=torch.from_numpy(np.array(edges)).long().t().contiguous()
edges_weight=[]
for i in range(ev.shape[0]):
    edges_weight+=[(int(ev[i,0]),int(ev[i,1]),ev[i,2])]
    

G.add_weighted_edges_from(edges_weight)

adj_bi=nx.adjacency_matrix(G,weight=None).todense()
adj_weight=nx.adjacency_matrix(G).todense()



data_list = [Data(...), ..., Data(...)]
loader = DataLoader(data_list, batch_size=32)


test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DenseDataLoader(test_dataset, batch_size=20)
val_loader = DenseDataLoader(val_dataset, batch_size=20)
train_loader = DenseDataLoader(train_dataset, batch_size=20)


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

class Net_el_w(torch.nn.Module):#GCN
    def __init__(self):
        super(Net_el_w, self).__init__()

        
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



class Net_diff_pool(torch.nn.Module):
    def __init__(self):
        super(Net_diff_pool, self).__init__()

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(1, 6, 100, gnn_k=1, add_loop=True)
        self.gnn1_embed = GNN(1, 6, 4,  gnn_k=1, add_loop=True)

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn2_pool = GNN(1 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(1 * 64, 64, 64)

        self.gnn3_embed = GNN(3 * 64, 64, 64)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, 6)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net0().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    loss_all = 0

    
     for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, link_loss, ent_loss = model(data.x, data.adj, data.mask)
        loss = F.nll_loss(output, data.y.view(-1)) + link_loss + ent_loss
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()
    correct = 0

     
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.adj, data.mask)[0].max(dim=1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


best_val_acc = test_acc = 0
for epoch in range(1, 151):
    train_loss = train(epoch)
    val_acc = test(val_loader)
    if val_acc > best_val_acc:
        test_acc = test(test_loader)
        best_val_acc = val_acc
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Val Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                     val_acc, test_acc))











class Cox_GNN(nn.Module):
	def __init__(self, In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, Pathway_Mask):
		super(Cox_GNN, self).__init__()
		self.tanh = nn.Tanh()
		self.pathway_mask = Pathway_Mask
		###gene layer --> pathway layer
		self.sc1 = nn.Linear(In_Nodes, Pathway_Nodes)
		###pathway layer --> hidden layer
		self.sc2 = nn.Linear(Pathway_Nodes, Hidden_Nodes)
		###hidden layer --> hidden layer 2
		self.sc3 = nn.Linear(Hidden_Nodes, Out_Nodes, bias=False)
		###hidden layer 2 + age --> Cox layer
		self.sc4 = nn.Linear(Out_Nodes+1, 1, bias = False)
		self.sc4.weight.data.uniform_(-0.001, 0.001)
		###randomly select a small sub-network
		self.do_m1 = torch.ones(Pathway_Nodes)
		self.do_m2 = torch.ones(Hidden_Nodes)
		###if gpu is being used
		if torch.cuda.is_available():
			self.do_m1 = self.do_m1.cuda()
			self.do_m2 = self.do_m2.cuda()
		###

	def forward(self, x_1, x_2):
		###force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
		self.sc1.weight.data = self.sc1.weight.data.mul(self.pathway_mask)
		x_1 = self.tanh(self.sc1(x_1))
		if self.training == True: ###construct a small sub-network for training only
			x_1 = x_1.mul(self.do_m1)
		x_1 = self.tanh(self.sc2(x_1))
		if self.training == True: ###construct a small sub-network for training only
			x_1 = x_1.mul(self.do_m2)
		x_1 = self.tanh(self.sc3(x_1))
		###combine age with hidden layer 2
		x_cat = torch.cat((x_1, x_2), 1)
		lin_pred = self.sc4(x_cat)
		
		return lin_pred

