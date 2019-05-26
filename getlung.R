options(stringsAsFactors = FALSE)
setwd("C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer")
library( openxlsx)
library(CONOR)
 

lf=list.files(pattern='*xlsx')
phel=list()
xl=list()
xl_unlabel=list()

affy=sort(c(1,48,31,32,56,38,39,44,17,23,53,12,43,18,33,42,50,2,28,55,16,10,51,34,3,13,8,25,19,46,14,54,35))


illu=sort(c(40,41,36,29,24))


agilent=sort(c(7,11,27,15))
platform=4
 


for(fi in c(1:length(lf))){
    if(  !(strsplit(lf[fi],"_")[[1]][1] %in% c(affy,illu,agilent)) ){
      next
    }
    print(fi)
    table1=read.xlsx(lf[fi],sheet=1 )
    table2=read.xlsx(lf[fi],sheet=2 )
    table3=read.xlsx(lf[fi],sheet=3 )#colnames are 'entrezid, sample1 sample2 ..'
    
    if(nrow(table1)!=nrow(table2)){
      print(fi)
      print('should be treated specially')
      next
      
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
    

    phenotab=cbind(table1[nomissing,c('Pat_ID','Pat_Overall_Survival_Months','Pat_Died')],
                   Sam_Name=table2[nomissing,'Sam_Name'])
    if(sum(!nomissing)==0 & dim(table3)[2]-1==nrow(table1)){
    #xtab=table3[,-1][,nomissing]
    #xtab=table3[,-1][ ,colnames(table3[,-1])%in%phenotab$Sam_Name ]
    #xtab=cbind(table3[,1],xtab)#the first column of xtab is the entrezid
      xtab=table3
      xl[[fi]]=xtab
      xl_unlabel[[fi]]=NULL 
      phel[[fi]]=phenotab
    }else{
    if(sum(!nomissing)==0 & dim(table3)[2]-1!=nrow(table1)){
      #xtab=table3[,-1][,nomissing]
      #xtab=table3[,-1][ ,colnames(table3[,-1])%in%phenotab$Sam_Name ]
      #xtab=cbind(table3[,1],xtab)#the first column of xtab is the entrezid
      print(fi)
      print('inconsistent with nomissing')
    }else{
    
    if(sum(nomissing)==0 ){
      xl[[fi]]=NULL
      xl_unlabel[[fi]]=table3
      phel[[fi]]=NULL
    }else{
      print(fi)
      print('in between!!')
      phel[[fi]]=phenotab
      xtab=table3[,-1][ ,nomissing&colnames(table3[,-1])%in%phenotab$Sam_Name ]
      xtab=cbind(table3[,1],xtab)
      xl[[fi]]=xtab
      if(sum(!(nomissing&colnames(table3[,-1])%in%phenotab$Sam_Name))!=0){
        x_unlabel_tab=table3[,-1][ ,!(nomissing&colnames(table3[,-1])%in%phenotab$Sam_Name) ]
        x_unlabel_tab=cbind(table3[,1],x_unlabel_tab)
        xl_unlabel[[fi]]=x_unlabel_tab
        
      }else{
        xl_unlabel[[fi]]=NULL
      }
    }
    }
}
    
    
    
}

######test platform sample size
sum=0
for(i in 1:length(xl )){
  if(!is.null(xl[[i]])){
    sum=sum+dim(xl[[i]])[2]
  }
  print(dim(xl[[i]]))
}


######

#platform=0

save(phel,xl,xl_unlabel,file=paste0('data_',platform,'.rda'))
load(paste0('data_',platform,'.rda'))
#build list,and remove those with genes < 10000

 
n_phel=list()
n_xl=list()
n_xl_unlabel=list()
unlabel_batch=c()
 
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
    n_phel[[i]]=cbind(phel[[i]],batch=i)#adding batch
    print(nrow(n_phel[[i]]))
  }
  if(!is.null(xl[[i]])){
    n_xl[[i]]=xl[[i]]
    print(ncol(xl[[i]]))
  }
  if(!is.null(xl_unlabel[[i]])){
    n_xl_unlabel[[i]]=xl_unlabel[[i]]
    unlabel_batch=c(unlabel_batch,rep(i,ncol(xl_unlabel[[i]])-1))
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
  j=j+1
}

for(i in 1:length(n_xl_coge)){
  
  print(nrow(n_phel_coge[[ i ]])) 
   
  print(ncol(n_xl_coge[[ i ]])) 
  
}

dfphe=do.call(rbind,n_phel_coge)
dfx=do.call(cbind,n_xl_coge)

n_xl_coge_unlabel=list()

 

for(i in 1:length(n_xl_unlabel)){
  if(is.null(n_xl_unlabel[[i]]) ){
    next
  }
  if(ncol(n_xl_unlabel[[i]])==1){
    next
  }
  print(i)
  tmp=n_xl_unlabel[[i]][as.numeric(n_xl_unlabel[[i]][,1])%in%common_id,]
  n_xl_coge_unlabel[[i]]=tmp[order(as.numeric(tmp[,1])),-1]
  
  
  
}



newlist=list()
j=1
for(i in 1:length(n_xl_coge_unlabel)){
  if(is.null(n_xl_coge_unlabel[[i]]) ){
    next
  }
  if(sum(!(common_id%in%rownames(n_xl_coge_unlabel[[i]])) )>0){
    next
  }
  print(ncol(n_xl_coge_unlabel[[i]]))
  newlist[[j]]=n_xl_coge_unlabel[[i]]
  j=j+1
}
dfx_unlabel=do.call(cbind,newlist) 
if(all(!duplicated(dfphe[,1]))==TRUE){
  print('noUdplicated label')
}


x1=dfx[,dfphe$batch %in% affy]
x2=dfx[,dfphe$batch %in% illu]
x3=dfx[,dfphe$batch %in% agilent]

 

r1=xpn(x2,x3,skip.match=TRUE)
x23=cbind(r1$x,r1$y)
r2=xpn(x1,x23,skip.match=TRUE)
x=cbind(r2$x,r2$y)



save(dfx,dfphe,dfx_unlabel,unlabel_batch,common_id,x1,x2,x3,x,file=paste0('finalx_',platform,'.rda'))


