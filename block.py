# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:08:57 2019

@author: Administrator
"""

 

class block_edge_conv(torch.nn.Module):#gcn
    def __init__(self,num_features,in_channels=1,out_channels=4,gnn_k=1,gnn_type=2,activation='leaky',res=False,jump=None):
        super(block_edge_conv, self).__init__()
        
        self.num_features = num_features    
        self.in_channels=in_channels
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
        self.gnn1= GNN(self.in_channels, self.out_channels, gnn_type=self.gnn_type,gnn_k=self.gnn_k,add_loop=True,jump=jump)
         
        self.lin1times1 = torch.nn.Linear(self.out_channels,1)
        
    def forward(self, x, edge_index,edge_weight=None):
        tempsize = x.size()
        x = self.gnn1(x, edge_index,edge_weight)
        x = x.view(-1,self.num_features,self.out_channels)
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
            
        self.gnn0_embed = GNN(self.in_channels, out_channels,gnn_k=gnn_k,gnn_type=gnn_type,add_loop=True,jump=jump)
        self.lin0_1times1 = torch.nn.Linear(self.out_channels,in_channels)
        
    def forward(self, x_input, adj, mask=None):
        print('input')
        print(x_input.size())
        print(self.gnn0_embed)
        x = self.gnn0_embed(x_input,adj=adj,mask=mask)
        print('after gnn0_embed')
        print(x.size())
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
        self.gnn1_pool = GNN(self.in_channels, out_clusters,gnn_k=gnn_k,gnn_type=gnn_type,add_loop=True,jump=jump)
        self.gnn1_embed = GNN(self.in_channels, out_channels,gnn_k=gnn_k,gnn_type=gnn_type,add_loop=True,jump=jump)

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
    def __init__(self,num_features,in_channels=1,out_channels=4,gnn_k=1,gnn_type=[2],activation='leaky',res=False,jump=None,depthlist=[1],nb=1,nodelist=[1],fc_exist=False):
        super(cell0, self).__init__()
         #gnn_type 15 ,16,20 corresponds to A,A**2,A**3,where A is the adjacency matrix
        self.num_features = num_features    
        
        self.out_channels=out_channels
        self.gnn_k=gnn_k
        self.gnn_type=gnn_type
        self.res = res
        
        self.fc=fc_edge(self.num_features , nodelist)
        self.nb=nb
        self.conv=[]
        self.depthlist=depthlist
        self.fc_exist=fc
        if activation=='leaky':
            self.activ=F.leaky_relu
        elif activation=='elu':
            self.activ=F.elu
        elif activation=='relu':
            self.activ=F.relu
        for j in range(nb):    
            blocklist=[block_edge_conv(num_features,in_channels=self.in_channels,out_channels=self.out_channels,\
                                       gnn_k=gnn_k[j],gnn_type=gnn_type[j],activation=activation,jump=jump)]
            for i in range(depthlist[j]-1):
                blocklist+=[block_edge_conv(num_features,in_channels=self.in_channels,out_channels=self.out_channels,\
                                       gnn_k=gnn_k,gnn_type=gnn_type[j],activation=activation,jump=jump)]
            self.conv.append(blocklist)

         
        
    def forward(self, x, edge_index,edge_weight=None):
        #the input size is (num_features*batch,1)
        output=[]
        if self.res==False:
            for j in range(self.nb):
                for i in range(self.depthlist[j]):
                    x = self.conv[j][i](x, edge_index, edge_weight)
                output.append(x)
        else:
            for j in range(self.nb):
                for i in range(self.depthlist[j]):
                    x0 = x 
                    x = self.conv[j][i](x, edge_index, edge_weight) + x0
                output.append(x)
        if  self.nb >1:
            x = torch.cat(tuple(output),dim=-1)
            
            x = x.mean(dim=-1)#here the size of x is (num_features*batch)
        #the input size is (num_features*batch,1),fc changes it to be batch,1
        if self.fc_exist==True: 
            x = x.view(-1,self.num_features)
            x = self.fc(x)
        else:
            x = x.view(x.size()[0],1)
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
        self.conv=[]
        self.depthlist=depthlist
        self.fc_exist=fc_exist
        if activation=='leaky':
            self.activ=F.leaky_relu
        elif activation=='elu':
            self.activ=F.elu
        elif activation=='relu':
            self.activ=F.relu
        for j in range(nb):    
            blocklist=[block_adj_embed(num_features,in_channels=self.in_channels,out_channels=self.out_channels,\
                                       gnn_k=gnn_k,gnn_type=gnn_type, activation=activation,jump=jump)]
            for i in range(1,depthlist[j]):
                blocklist+=[block_adj_embed(num_features,in_channels=self.in_channels,out_channels=self.out_channels,\
                                       gnn_k=gnn_k,gnn_type=gnn_type, activation=activation,jump=jump)]
            blocklist[-1].reduce=reduce_tail
            self.conv.append(blocklist)

         
        
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
                    x = self.conv[j][i](x, adj[j], mask)
                output.append(x)
        else:
            for j in range(self.nb):
                for i in range(self.depthlist[j]):
                    x0 = x_input 
                    x = self.conv[j][i](x, adj[j], mask) + x0
                output.append(x)
         
        if  self.nb >1:
            x = torch.cat(tuple(output),dim=-1)
            x = x.mean(dim=-1)#here the size of x is (num_features*batch)
        #the input size is (num_features*batch,1),fc changes it to be batch,1
         
        if self.fc_exist==True: 
            x = x.view(-1,self.num_features)
            x = self.fc(x)
        else:
             
            x = x.view(*(list(x.size()[0:2])+[-1]))
            
        return x



    
class Netadj(torch.nn.Module):
    #note that for gnn2_pool,the input dimension is the number of hidden nodes after the first pooling operation
    def __init__(self,num_features ,out_channels=4,out_clusters_list=[40,20],gnn_k=2,gnn_type=1,\
                 activation='leaky',res=False,jump=None,depthlist=[2,2],fc_node_list=[20,10,1]):
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
        #out_clusters_list,the size of the num_features or number of the clusters
         
         
        embed_list=[cell1(num_features,in_channels=1,out_channels=self.out_channels,\
            gnn_type=gnn_type,gnn_k=gnn_k,res=res,jump=jump,depthlist=depthlist,nb=len(depthlist),fc_exist=False,reduce_tail=False) ]
        for i in range(len(out_clusters_list)-1):
            embed_list=embed_list+[ cell1(self.out_clusters_list[i-1],in_channels=self.out_channels,out_channels=self.out_channels,\
            gnn_type=gnn_type,gnn_k=gnn_k,res=res,jump=jump,depthlist=depthlist,nb=len(depthlist),fc_exist=False,reduce_tail=False) ]
         
        pool_list=[cell1(num_features,in_channels=1,out_channels=out_clusters_list[0],\
            gnn_type=gnn_type,gnn_k=1,res=res,jump=jump,depthlist=[1],nb=1,fc_exist=False,reduce_tail=False)]
        for i in  range(1,len(self.out_clusters_list)):
            pool_list=pool_list+[cell1(self.out_clusters_list[i-1],in_channels=self.out_channels,out_channels=self.out_clusters_list[i],\
            gnn_type=gnn_type,gnn_k=1,res=res,jump=jump,depthlist=[1],nb=1,fc_exist=False,reduce_tail=False)]
         
        self.embed_list=embed_list
        self.pool_list=pool_list
        self.postfc= fc_edge(out_clusters_list[-1],extra_nodelist=fc_node_list)
        self.lin1times1 = torch.nn.Linear(self.out_channels,1)
        
    

    def forward(self, x, adj, mask=None):
        #print('forward diff')
        l_loss=0
        e_loss=0
        for i in range(len(self.out_clusters_list)):
            #print(x.size())    
            print('layer')
            print(i)
            print(x.size())
            s = self.pool_list[i](x, adj=adj, mask=mask)
            print('pool finished')
            x = self.embed_list[i](x, adj=adj, mask=mask)
             
            x, adj, l1, e1 = dense_diff_pool(x, adj[0], s, mask)
            #adj=[adj[0],adj[0]**2,adj[0]**3]
            print('after diff pool')
            print(x.size())
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
        x = self.lin1times1(x)
         
        x = x.view(x.size()[0:2])
         
        x = self.postfc(x)
        return x, l_loss, e_loss




model=Netadj(num_features)
model.pool_list[0].conv[0][0](x_train,adj_train[0]).size()

x,l_loss,e_loss=model(x_train,adj_train)
    
    
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

model = cell1(num_features,in_channels=1,out_channels=4,gnn_type=[1,1],gnn_k=2,res=True,jump='lstm',depthlist=[2,3],nb=2,fc_exist=True).to(device)
#model(x_edge_train,edge_index_train).size()==x_edge_train.size()

model(x_adj,x_adj_adj).size()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
ret=do_c(epoch, y_train, censor_train, y_test, censor_test,process_type,train_loader,test_loader,x_train,x_test,adj_train,adj_test,optimizer,device)
        
