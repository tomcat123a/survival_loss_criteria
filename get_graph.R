#library(igraph)
options(stringsAsFactors = FALSE)

library(data.table)



load('C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/finalx_0.rda')#get dfx,dfphe
load('C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/finalx_unlabel_0.rda')
lungnet=fread('C:/Users/Administrator/Desktop/deepgraphsurv/lung/lung_top/lung_top',fill=TRUE)
lungnetvec=as.vector(lungnet)

setwd("C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/survdata")
dt=lungnet
rownames(dfx )=common_id
rownames(dfx_unlabel)=common_id

#setwd("C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/survdata")
for(i in 0:4){
  v1=apply(dfx,1,function(x){
    var(x)
  })
  v2=apply(dfx,1,function(x){
    var(x) /(abs(mean(x)) + 0.00001  )})
  
  index1=which(rank(v1)>(0.8-0.1*i)*length(v1) & rank(v2)>(0.8-0.1*i)*length(v2))
  length(index1)
  
  filtered_dfx=dfx[index1,]
  
  write.csv(filtered_dfx,paste0('lung_x_',i,'.csv'))
  
  write.csv(dfphe[,2],paste0('lung_t_',i,'.csv'))
  
  write.csv(dfphe[,3],paste0('lung_c_',i,'.csv'))
  
  write.csv(dfx_unlabel[index1,],paste0('lung_un_',i,'.csv'))
  
  name0=rownames(filtered_dfx)
  n_name0=as.numeric(name0)
  dt1=dt[dt$V1 %in% n_name0,]
  dt2=dt1[dt1$V2 %in% n_name0,]
  write.csv(dt2,file=paste0('dt_',i,'.csv'))
}
#generate toy dataset
for(i in 5:5){
  v1=apply(dfx,1,function(x){
    var(x)
  })
  v2=apply(dfx,1,function(x){
    var(x) /(abs(mean(x)) + 0.00001  )})
  
  index1=which(rank(v1)>(0.97)*length(v1) & rank(v2)>(0.97)*length(v2))
  length(index1)
  
  filtered_dfx=dfx[index1,]
  
  write.csv(filtered_dfx,paste0('lung_x_',i,'.csv'))
  
  write.csv(dfphe[,2],paste0('lung_t_',i,'.csv'))
  
  write.csv(dfphe[,3],paste0('lung_c_',i,'.csv'))
  
  write.csv(dfx_unlabel[index1,],paste0('lung_un_',i,'.csv'))
  
  name0=rownames(filtered_dfx)
  n_name0=as.numeric(name0)
  dt1=dt[dt$V1 %in% n_name0,]
  dt2=dt1[dt1$V2 %in% n_name0,]
  write.csv(dt2,file=paste0('dt_',i,'.csv'))
}



#############get data for unlabel

load('C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/finalx_unlabel_0.rda')




setwd("C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/survdata")
for(i in 0:4){
  v1=apply(dfx_unlabel,1,function(x){
    var(x)
  })
  v2=apply(dfx_unlabel,1,function(x){
    var(x) /(abs(mean(x)) + 0.001  )})
  
  index1=which(rank(v1)>(0.8-0.1*i)*length(v1) & rank(v2)>(0.8-0.1*i)*length(v2))
  length(index1)
  
  filtered_dfx=dfx_unlabel[index1,]
  
  write.csv(filtered_dfx,paste0('lung_x_unlabel_',i,'.csv'))
  
  #write.csv(dfphe[,2],paste0('lung_t_',i,'.csv'))
  
  #write.csv(dfphe[,3],paste0('lung_c_',i,'.csv'))
  name0=rownames(filtered_dfx)
  #n_name0=as.numeric(name0)
  #dt1=dt[dt$V1 %in% n_name0,]
  #dt2=dt1[dt1$V2 %in% n_name0,]
  #write.csv(dt2,file=paste0('dt_',i,'.csv'))
}









##################

