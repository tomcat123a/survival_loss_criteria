# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:08:57 2019

@author: Administrator
"""

 

class block_edge_conv(torch.nn.Module):#gcn
    def __init__(self,num_features,hidden_channels=4,out_channels=4,gnn_k=1,gnn_type=2,activation='leaky',res=False,jump=None):
        super(block_edge_conv, self).__init__()
        
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
        if jump=='lstm' and out_channels%2==1 and gnn_k%2==1:
            raise ValueError('out_channels%2==1 and gnn_k%2==1')
        self.gnn1= GNN(1, self.hidden_channels, self.out_channels, gnn_type=self.gnn_type,gnn_k=self.gnn_k,add_loop=True,jump=jump)
         
        self.lin1times1 = torch.nn.Linear(self.out_channels,1)
        
    def forward(self, x, edge_index,edge_weight=None):
        tempsize = x.size()
        if edge_weight is None:
            x = self.gnn1(x, edge_index)
        else:
            x = self.gnn1(x, edge_index,edge_weight)
        x = x.view(-1,self.num_features,self.out_channels)
        x = self.activ(self.lin1times1(x))
        x = x.view(tempsize)
        return x



class fc_edge(torch.nn.Module):#gcn
    def __init__(self,num_features,nodelist=[1],activation='leaky',res=False):
        super(fc_edge, self).__init__()
        
        self.num_features = num_features    
        
        if activation=='leaky':
            self.activ=nn.LeakyReLU()
        elif activation=='elu':
            self.activ=nn.ELU()
        elif activation=='relu':
            self.activ=nn.ReLU()
         
        if len(nodelist)==1:
               
            self.lin1 = torch.nn.Linear(self.num_features, 1) 
        else:
            #print(len(nodelist))
            li=[torch.nn.Linear(self.num_features, nodelist[0]),torch.nn.BatchNorm1d(nodelist[0]),\
                self.activ,nn.Dropout()]
            for i in range(1,len(nodelist)-1): 
                li = li+[torch.nn.Linear(nodelist[i-1], nodelist[i]),torch.nn.BatchNorm1d(nodelist[i]),\
                self.activ,nn.Dropout()]
            li = li + [torch.nn.Linear(nodelist[-2], 1)]
            self.lin1 = nn.Sequential(*li)
        
        
    def forward(self, x ):
        
        x = x.view(x.size()[0:2])
        x = self.lin1(x)
        #x = self.lin2(x)
        return x



class block_adj_embed(torch.nn.Module):
    #note that for gnn2_pool,the input dimension is the number of hidden nodes after the first pooling operation
    def __init__(self,num_features,hidden_channels=50,out_channels=50,gnn_k=1,gnn_type=1,activation='leaky',res=False,jump=None\
                 ):
        super(block_adj_embed, self).__init__()
        self.num_features = num_features    
         
        self.out_channels=out_channels
         
        self.hidden_channels=hidden_channels
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
            
        self.gnn0_embed = GNN(1, hidden_channels, out_channels,gnn_k=gnn_k,gnn_type=gnn_type,add_loop=True,jump=jump)
        self.lin0_1times1 = torch.nn.Linear(self.out_channels,1)
        
    def forward(self, x_input, adj, mask=None):
        x = self.gnn0_embed(x_input,adj=adj,mask=mask)
        x = self.lin0_1times1(x)
        return x
    
class block_adj_pool(torch.nn.Module):
    #note that for gnn2_pool,the input dimension is the number of hidden nodes after the first pooling operation
    def __init__(self,num_features,hidden_channels=10,out_channels=10,out_clusters=20,gnn_k=1,gnn_type=1,activation='leaky',res=False,jump=None\
                 ,shrink_factor=0.2):
        super(block_adj_pool, self).__init__()
        self.num_features = num_features    
        self.out_clusters=out_clusters
        self.out_channels=out_channels
        self.sh_f=shrink_factor
        self.hidden_channels=hidden_channels
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
        self.gnn1_pool = GNN(1, out_clusters, out_clusters,gnn_k=gnn_k,gnn_type=gnn_type,add_loop=True,jump=jump)
        self.gnn1_embed = GNN(1, hidden_channels, out_channels,gnn_k=gnn_k,gnn_type=gnn_type,add_loop=True,jump=jump)

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
# a2(x_weight,x_weight_index,x_weight_weight)
# 
# a3=block_adj_embed(num_features,gnn_k=3,gnn_type=1,activation='leaky',jump='lstm')
# a3(x_adj,x_adj_adj).size()
# 
# a4=block_adj_pool(num_features,gnn_k=3,gnn_type=1,activation='leaky',jump='lstm')
# x4,l4,e4,adj4=a4(x_adj,x_adj_adj)
# a5=fc_edge(88,[8,1])
# a5(torch.ones(5,88))
# =============================================================================


class Net0(torch.nn.Module):#gcn
    def __init__(self,num_features,hidden_channels=4,out_channels=4,gnn_k=1,gnn_type=2,activation='leaky',res=False,jump=None,n=1):
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
            
        blocklist=[]
        for i in range(n):
            blocklist+=[block_edge_conv(num_features,hidden_channels=out_channels,out_channels=out_channels,\
                                   gnn_k=gnn_k,gnn_type=gnn_type,activation=activation,jump=jump)]
        #self.gnn2 = GNN(100, 64, 64, gnn_type=2,add_loop=True)

        #self.bn1 = torch.nn.BatchNorm1d(self.hidden_channels)
        self.conv=nn.Sequential(*blocklist)
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
