options(stringsAsFactors = FALSE)
setwd("C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer")
library( openxlsx)
library(CONOR)
  

lf=list.files(pattern='*xlsx')
phel=list()
xl=list()
xl_unlabel=list()
unlabel_batch=list()
affy=sort(c(1,48,31,32,56,38,39,44,17,23,53,12,43,18,33,42,50,2,28,55,16,10,51,34,3,13,8,25,19,46,14,54,35))


illu=sort(c(40,41,36,29,24))


agilent=sort(c(7,11,27,15))
platform=4
 

for(fi in 1:length(lf)){
  batchid =strsplit(lf[fi],"_")[[1]][1]
    if(  !( batchid %in% c(affy,illu,agilent)) ){
      next
    }
    print(fi)
    table1=read.xlsx(lf[fi],sheet=1 )
    table2=read.xlsx(lf[fi],sheet=2 )
    table3=read.xlsx(lf[fi],sheet=3 )#colnames are 'entrezid, sample1 sample2 ..'
    
    if(nrow(table1)!=nrow(table2)){
      print(fi)
      print('should be treated specially')
       
      
      common_pat=intersect(table1[,1],table2[,1]) 
      if(length(common_pat)==0){
        print(fi)
        print('should be treated specially')
        next
      }
      table1=table1[table1[,1]%in%common_pat,]
      table2=table2[table2[,1]%in%common_pat & !duplicated(table2[,1]) ,]
      if(dim(table1)[1]!=dim(table2)[1]){
        print(fi)
        print('dim error!')
      }
    }
    #recompute nomissing
    nomissing=complete.cases( table1['Pat_Overall_Survival_Months'] )&complete.cases( table1['Pat_Died'] )
    

    
    if(all(nomissing) & dim(table3)[2]-1==nrow(table1)){
    #xtab=table3[,-1][,nomissing]
    #xtab=table3[,-1][ ,colnames(table3[,-1])%in%phenotab$Sam_Name ]
    #xtab=cbind(table3[,1],xtab)#the first column of xtab is the entrezid
      phenotab=cbind(table1[nomissing,c('Pat_ID','Pat_Overall_Survival_Months','Pat_Died')],
                     Sam_Name=table2[nomissing,'Sam_Name'],batch=batchid)
      xtab=table3
      xl[[fi]]=xtab
      xl_unlabel[[fi]]=NULL 
      phel[[fi]]=phenotab
      unlabel_batch[[fi]]=NULL
    }else{
    if(all( nomissing) & dim(table3)[2]-1!=nrow(table1)){
      #xtab=table3[,-1][,nomissing]
      #xtab=table3[,-1][ ,colnames(table3[,-1])%in%phenotab$Sam_Name ]
      #xtab=cbind(table3[,1],xtab)#the first column of xtab is the entrezid
      print(fi)
      print('inconsistent with nomissing')
    }else{
    
    if(all(!nomissing)  ){
      xl[[fi]]=NULL
      xl_unlabel[[fi]]=table3
      phel[[fi]]=NULL
      unlabel_batch[[fi]]=rep(batchid,ncol(xl_unlabel[[fi]])-1)
    }else{
      print(fi)
      print('in between!!')
      phenotab=cbind(table1[nomissing,c('Pat_ID','Pat_Overall_Survival_Months','Pat_Died')],
                     Sam_Name=table2[nomissing,'Sam_Name'],batch=batchid)
      phel[[fi]]=phenotab
      table3data=table3[,-1]
      if(sum(duplicated(colnames(table3data)))>0){
        print(fi)
        print('warning! duplicated')
      }
      
      labindex=which( colnames(table3data)%in%( phenotab$Sam_Name  )   )
      
      unlabindex=which( !(colnames(table3data)%in%( phenotab$Sam_Name  ))   )
       
      xtab=table3data[ ,labindex]
      xtab=cbind(table3[,1],xtab)
      
      xl[[fi]]=xtab
      if(length(unlabindex)>0){
        x_unlabel_tab=table3data[ ,unlabindex]
        x_unlabel_tab=cbind(table3[,1],x_unlabel_tab)
        xl_unlabel[[fi]]=x_unlabel_tab
        unlabel_batch[[fi]]=rep(batchid,ncol(xl_unlabel[[fi]])-1)
        
      }else{
        xl_unlabel[[fi]]=NULL
        unlabel_batch[[fi]]=NULL
      }
    }
    }
}
    
    
    
}

######test platform sample size
 


######

#platform=0

save(phel,xl,xl_unlabel,unlabel_batch,affy,illu,agilent,file=paste0('data_',platform,'.rda'))
load(paste0('data_',platform,'.rda'))
#build list,and remove those with genes < 10000

 
n_phel=list()
n_xl=list()
n_xl_unlabel=list()
n_unlabel_batch=list()
 
for(i in 1:length(xl)){
  print(i)
  if(!is.null(xl[[i]])  ){
    if(dim(xl[[i]])[1]<10000){
      next
    }
  }
  if(!is.null(xl_unlabel[[i]])  ){
    if(dim(xl_unlabel[[i]])[1]<10000){
      next
    }
  }
  
  if(NA%in%xl[[i]][,2]){
    print('na detected!!!')
    next
  }
  if(length(xl[[i]]) + length(xl_unlabel[[i]])==0 ){
    next
  }
   

  
  if(!is.null(phel[[i]])){
    n_phel[[i]]= phel[[i]] 
    print(nrow(n_phel[[i]]))
  }
  if(!is.null(xl[[i]])){
    n_xl[[i]]=xl[[i]]
    print(ncol(xl[[i]]))
  }
  if(!is.null(xl_unlabel[[i]])){
    n_xl_unlabel[[i]]=xl_unlabel[[i]]
    n_unlabel_batch[[i]]=unlabel_batch[[i]]
  }
   #print(nrow(n_phel[[i]]))

}


#build a dataframe that cancatenates n_phel
#for(i in 1:length(n_phel)){
#  if(i==1){
#    dfphe=n_phel[[1]]
#  }else{
#    dfphe=rbind(dfphe,n_phel[[i]])
#  }
#}


#remove duplication of samples




#getcommon entrezid
for(i in 1:length(n_xl)){
  if(i==1){
      name0=n_xl[[i]][,1]
  }else{
    if(!is.null(n_xl[[i]])){
      name0=intersect(name0,n_xl[[i]][,1])
    }
  }
}
common_id=sort(as.numeric(name0))#11978
#filter n_xl for common genes
#n_xl[,1] is the gene entrezid
n_xl_coge=list()
n_phel_coge=list()
 
j=1
for(i in 1:length(n_xl)){
  if(is.null(n_xl[[i]]) ){
     
    next
  }
  
  n_phel_coge[[j]]=n_phel[[i]]
  tmp=n_xl[[i]][as.numeric(n_xl[[i]][,1])%in%common_id,]
   
  n_xl_coge[[j]]=tmp[order(as.numeric(tmp[,1])),-1]
  
  rownames(n_xl_coge[[j]])=common_id
  j=j+1
}


dfphe=do.call(rbind,n_phel_coge)
dfx=do.call(cbind,n_xl_coge)

n_xl_coge_unlabel=list()
n_batch_coge_unlabel=list()
 

for(i in 1:length(n_xl_unlabel)){
  if(is.null(n_xl_unlabel[[i]]) ){
    next
  }
  if(ncol(n_xl_unlabel[[i]])==1){
    next
  }
  print(i)
  tmp=n_xl_unlabel[[i]][as.numeric(n_xl_unlabel[[i]][,1])%in%common_id,]
  rnames=tmp[,1]
  n_xl_coge_unlabel[[i]]=tmp[order(as.numeric(tmp[,1])),-1]
  rownames(n_xl_coge_unlabel[[i]])=rnames
  n_batch_coge_unlabel[[i]]=n_unlabel_batch[[i]]
  
}



newlist=list()
newlist1=list()
j=1
for(i in 1:length(n_xl_coge_unlabel)){
  if(is.null(n_xl_coge_unlabel[[i]]) ){
    next
  }
  if(!all(common_id%in%rownames(n_xl_coge_unlabel[[i]])) ){
    next
  }
  print(ncol(n_xl_coge_unlabel[[i]]))
  newlist[[j]]=n_xl_coge_unlabel[[i]]
  newlist1[[j]]=n_batch_coge_unlabel[[i]]
  j=j+1
}

dfx_unlabel=do.call(cbind,newlist) 
dfx_unlabel_batch=unlist(newlist1) 

if(all(!duplicated(dfphe[,1]))==TRUE){
  print('noUdplicated label')
}

save(dfx,dfphe,dfx_unlabel,dfx_unlabel_batch,common_id,affy,illu,agilent,file=paste0('finalx_start_',platform,'.rda'))

platform=4
load(paste0('finalx_start_',platform,'.rda'))
save(affy,illu,agilent,file=paste0('finalx_plat_',platform,'.rda'))


x1=dfx[,dfphe$batch %in% affy]
x2=dfx[,dfphe$batch %in% illu]
x3=dfx[,dfphe$batch %in% agilent]


 


r1=xpn(x2,x3,skip.match=TRUE)
x23=cbind(r1$x,r1$y)
r2=xpn(x1,x23,skip.match=TRUE)
x=cbind(r2$x,r2$y)

 
ux=xpn(z1,z2)


save(dfx,dfphe,dfx_unlabel,unlabel_batch,common_id,x1,x2,x3,x,ux,file=paste0('finalx_',platform,'.rda'))









