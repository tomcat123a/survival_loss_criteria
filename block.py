# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:08:57 2019

@author: Administrator
"""

 

class block_edge_conv(torch.nn.Module):#gcn
    def __init__(self,num_features,out_channels=4,gnn_k=1,gnn_type=2,activation='leaky',res=False,jump=None):
        super(block_edge_conv, self).__init__()
        
        self.num_features = num_features    
        
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
        self.gnn1= GNN(1, self.out_channels, gnn_type=self.gnn_type,gnn_k=self.gnn_k,add_loop=True,jump=jump)
         
        self.lin1times1 = torch.nn.Linear(self.out_channels,1)
        
    def forward(self, x, edge_index,edge_weight=None):
        tempsize = x.size()
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
            self.activ=[nn.LeakyReLU()]*len(nodelist)
        elif activation=='elu':
            self.activ=[nn.ELU()]*len(nodelist)
        elif activation=='relu':
            self.activ=[nn.ReLU()]*len(nodelist)
         
        if len(nodelist)==1:
               
            self.lin1 = torch.nn.Linear(self.num_features, 1) 
        else:
            #print(len(nodelist))
            li=[torch.nn.Linear(self.num_features, nodelist[0]),torch.nn.BatchNorm1d(nodelist[0]),\
                self.activ[0],nn.Dropout()]
            for i in range(1,len(nodelist)-1): 
                li = li+[torch.nn.Linear(nodelist[i-1], nodelist[i]),torch.nn.BatchNorm1d(nodelist[i]),\
                self.activ[i],nn.Dropout()]
            li = li + [torch.nn.Linear(nodelist[-2], 1)]
            self.lin1 = nn.Sequential(*li)
        
        
    def forward(self, x ):
        
        x = x.view(x.size()[0:2])
        x = self.lin1(x)
        #x = self.lin2(x)
        return x



class block_adj_embed(torch.nn.Module):
    #note that for gnn2_pool,the input dimension is the number of hidden nodes after the first pooling operation
    def __init__(self,num_features,out_channels=50,gnn_k=1,gnn_type=1,activation='leaky',res=False,jump=None\
                 ):
        super(block_adj_embed, self).__init__()
        self.num_features = num_features    
         
        self.out_channels=out_channels
         
        
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
            
        self.gnn0_embed = GNN(1, out_channels,gnn_k=gnn_k,gnn_type=gnn_type,add_loop=True,jump=jump)
        self.lin0_1times1 = torch.nn.Linear(self.out_channels,1)
        
    def forward(self, x_input, adj, mask=None):
        x = self.gnn0_embed(x_input,adj=adj,mask=mask)
        x = self.lin0_1times1(x)
        return x
    
class block_adj_pool(torch.nn.Module):
    #note that for gnn2_pool,the input dimension is the number of hidden nodes after the first pooling operation
    def __init__(self,num_features,out_channels=10,out_clusters=20,gnn_k=1,gnn_type=1,activation='leaky',res=False,jump=None\
                 ,shrink_factor=0.2):
        super(block_adj_pool, self).__init__()
        self.num_features = num_features    
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
        self.gnn1_pool = GNN(1, out_clusters,gnn_k=gnn_k,gnn_type=gnn_type,add_loop=True,jump=jump)
        self.gnn1_embed = GNN(1, out_channels,gnn_k=gnn_k,gnn_type=gnn_type,add_loop=True,jump=jump)

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
    def __init__(self,num_features,out_channels=[4],gnn_k=[1],gnn_type=[2],activation='leaky',res=False,jump=None,nlist=[1],nb=1,nodelist=[1]):
        super(cell0, self).__init__()
         #gnn_type 15 ,16,20 corresponds to A,A**2,A**3,where A is the adjacency matrix
        self.num_features = num_features    
        
        self.out_channels=out_channels
        self.gnn_k=gnn_k
        self.gnn_type=gnn_type
        self.res = res
        self.depthcat=nn.Linear(sum(out_channels),1)
        self.fc=fc_edge(self.num_features , nodelist)
        self.nb=nb
        self.conv=[]
        self.nlist=nlist
        if activation=='leaky':
            self.activ=F.leaky_relu
        elif activation=='elu':
            self.activ=F.elu
        elif activation=='relu':
            self.activ=F.relu
        for j in range(nb):    
            blocklist=[]
            for i in range(nlist[j]):
                blocklist+=[block_edge_conv(num_features,out_channels=out_channels[j],\
                                       gnn_k=gnn_k[j],gnn_type=gnn_type[j],activation=activation,jump=jump)]
            self.conv.append(blocklist)

         
        
    def forward(self, x, edge_index,edge_weight=None):
        output=[]
        if self.res==False:
            for j in range(self.nb):
                for i in range(self.nlist[j]):
                    x = self.conv[j][i](x, edge_index, edge_weight)
                output.append(x)
        else:
            for j in range(self.nb):
                for i in range(self.nlist[j]):
                    x0 = x 
                    x = self.conv[j][i](x, edge_index, edge_weight) + x0
                output.append(x)
        x = torch.cat(tuple(output),dim=-1)
        
        x = x.mean(dim=-1)
        x = x.view(-1,self.num_features)
        x = self.fc(x)
        return x


    
class Net_diff_pool0(torch.nn.Module):
    #note that for gnn2_pool,the input dimension is the number of hidden nodes after the first pooling operation
    def __init__(self,num_features ,out_channels=50,out_clusters_list=[200],gnn_k=1,gnn_type=1,activation='leaky',res=False,jump=None\
                 ):
        super(Net_diff_pool0, self).__init__()
        self.num_features = num_features    
        self.out_clusters=out_clusters
        self.out_channels=out_channels
        #out_clusters_list,the size of the num_features or number of the clusters
         
        if activation=='leaky':
            self.activ=F.leaky_relu
        elif activation=='elu':
            self.activ=F.elu
        elif activation=='relu':
            self.activ=F.relu
        embed_list=[]
        pool_list=[]
        for i in out_clusters_list:
            pool_list=pool_list+[GNN(1 ,out_clusters,gnn_k=gnn_k,gnn_type=1,add_loop=True,res=res,jump=jump)]
         
        self.gnn1_embed = GNN(1 , out_channels,gnn_k=gnn_k,gnn_type=1,add_loop=True,res=res,jump=jump)

        
        self.gnn2_pool = GNN(in_channels=out_channels,hidden_channels=int(out_clusters*self.sh_f),out_channels=int(out_clusters*self.sh_f),gnn_k=gnn_k,gnn_type=1,add_loop=True,res=res,jump=jump)
        self.gnn2_embed = GNN(out_channels, out_channels, out_channels,gnn_k=gnn_k,gnn_type=1,add_loop=True,res=res,jump=jump)

        self.gnn3_embed = GNN(out_channels, out_channels, out_channels,gnn_k=gnn_k,gnn_type=1,add_loop=True,res=res,jump=jump)

        self.lin1 = torch.nn.Linear(int(self.sh_f*self.out_clusters), 1)
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
        x = x.view(-1,int(self.sh_f*self.out_clusters),self.out_channels)
        #x = x.mean(dim=-1)#1*1 convolution
        x = self.activ(self.lin1times1(x))
        x = x.view(x.size()[0:2])
        
        x = self.activ(self.lin1(x))
        
        return x, l1 + l2, e1 + e2






    
    
model = cell0(num_features,out_channels=[n_channels],gnn_type=[gnn_type],gnn_k=[gnn_k],res=False,jump=jump_type,nb=1).to(device)
model(x_edge_train,edge_index_train).size()==x_edge_train.size()
model(x_weight,x_weight_index,x_weight_weight).size()==x_weight.size()


model = Net0(num_features,out_channels=n_channels,gnn_type=14,gnn_k=gnn_k,res=False,jump=jump_type,n=2).to(device)
model(x_weight,x_weight_index,x_weight_weight).size()==x_weight.size()

model = Net0(num_features,out_channels=n_channels,gnn_type=gnn_type,gnn_k=4,res=False,jump=jump_type,n=2).to(device)
model(x_edge_train,edge_index_train).size()==x_edge_train.size()

model = Net0(num_features,out_channels=n_channels,gnn_type=16,gnn_k=gnn_k,res=True,jump='max',n=2).to(device)
model(x_edge_train,edge_index_train).size()==x_edge_train.size()

model = Net0(num_features,out_channels=n_channels,gnn_type=7,gnn_k=gnn_k,res=True,jump=None,n=2).to(device)
model(x_edge_train,edge_index_train).size()==x_edge_train.size()

model = cell0(num_features,out_channels=[n_channels],gnn_type=[14],gnn_k=[gnn_k],res=True,jump=jump_type,nb=1,nodelist=[3,2,1]).to(device)
model(x_edge_train,edge_index_train).size()==torch.Size([int(x_edge_train.size()[0]/num_features),1])

model(x_weight,x_weight_index,x_weight_weight).size()==torch.Size([int(x_edge_train.size()[0]/num_features),1])


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
ret=do_c(epoch, y_train, censor_train, y_test, censor_test,process_type,train_loader,test_loader,x_train,x_test,adj_train,adj_test,optimizer,device)
        
