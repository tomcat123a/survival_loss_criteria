options(stringsAsFactors = FALSE)
setwd("C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer")
library( openxlsx)
lf=list.files(pattern='*xlsx')
phel=list()
xl=list()
for(fi in c(1:length(lf))){
    print(fi)
    table1=read.xlsx(lf[fi],sheet=1 )
    table2=read.xlsx(lf[fi],sheet=2 )
    table3=read.xlsx(lf[fi],sheet=3 )
    nomissing=complete.cases( table1['Pat_Overall_Survival_Months'] )&complete.cases( table1['Pat_Died'] )
    if(sum(nomissing)==0){
      next
    }
    if(nrow(table1)!=nrow(table2)){
      print(fi)
      print('should be treated specially')
      next
    }


    phenotab=cbind(table1[nomissing,c('Pat_ID','Pat_Overall_Survival_Months','Pat_Died')],
                   Sam_Name=table2[nomissing,'Sam_Name'])
    xtab=table3[,-1][,nomissing]
    xtab=cbind(table3[,1],xtab)#the first column of xtab is the entrezid
    phel[[fi]]=phenotab
    xl[[fi]]=xtab
}



length(phel)

save(phel,xl,file='data.rda')
load('data.rda')
#build list,and remove those with genes < 10000

id=c()
n_phel=list()
n_xl=list()
j=1
for(i in 1:length(xl)){
  if(length(xl[[i]])==0){
    next
  }
  if(dim(xl[[i]])[1]<10000){
    next
  }
  if(NA%in%xl[[i]][,2]){
    next
  }
  id=c(id,i)

  n_phel[[j]]=phel[[i]]
  n_xl[[j]]=xl[[i]]
  j=j+1

}



#build a dataframe that cancatenates n_phel
#for(i in 1:length(n_phel)){
#  if(i==1){
#    dfphe=n_phel[[1]]
#  }else{
#    dfphe=rbind(dfphe,n_phel[[i]])
#  }
#}


dfphe=do.call(rbind,n_phel)
#remove duplication of samples




#getcommon entrezid
for(i in 1:length(n_xl)){
  if(i==1){
      name0=n_xl[[i]][,1]
  }else{
    name0=intersect(name0,n_xl[[i]][,1])
  }
}
common_id=sort(as.numeric(name0))
#filter n_xl for common genes
#n_xl[,1] is the gene entrezid
n_xl_coge=list()
for(i in 1:length(n_xl)){
    tmp=n_xl[[i]][n_xl[[i]][,1]%in%name0,]
    n_xl_coge[[i]]=tmp[order(tmp[,1]),-1]
}




dfx=do.call(cbind,n_xl_coge)

rownames(dfx)=common_id

if(all(!duplicated(dfphe[,1]))==TRUE){
  print('noUdplicated label')
}

save(dfx,dfphe,common_id,file='finalx_0.rda')



##################################
#get unlabled expression
setwd("C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer")
lf=list.files(pattern='*xlsx')
phel_unlabel=list()
xl_unlabel=list()
for(fi in c(1:length(lf))){
  print(fi)
  table1=read.xlsx(lf[fi],sheet=1 )
  table2=read.xlsx(lf[fi],sheet=2 )
  
  nomissing=!(complete.cases( table1['Pat_Overall_Survival_Months'] )&complete.cases( table1['Pat_Died'] ))
  
  if(sum(nomissing)==0){
    next
  }
  if(nrow(table1)!=nrow(table2)){
    print(fi)
    print('should be treated specially')
    next
  }
  phenotab=cbind(table1[nomissing,c('Pat_ID')],
                 Sam_Name=table2[nomissing,'Sam_Name'])
  
  phel_unlabel[[fi]]=phenotab
  
  table3=read.xlsx(lf[fi],sheet=3 )
  xtab=table3[,-1][,nomissing]
  xtab=cbind(table3[,1],xtab)
  
  xl_unlabel[[fi]]=xtab
}



save(xl_unlabel,phel_unlabel,file='data_unlabel.rda')
#load('data.rda')
#build list,and remove those with genes < 10000

id=c()
#n_phel=list()
n_xl_unlabel=list()
n_phel_unlabel=list()
j=1
for(i in 1:length(xl_unlabel)){
  if(length(xl_unlabel[[i]])==0){
    next
  }
  if(dim(xl_unlabel[[i]])[1]<10000){
    next
  }
  if(NA%in%xl_unlabel[[i]][,2]){
    next
  }
  id=c(id,i)
  
  n_phel_unlabel[[j]]=phel_unlabel[[i]]
  n_xl_unlabel[[j]]=xl_unlabel[[i]]
  j=j+1
  
}


dfphe_unlabel=do.call(rbind,n_phel_unlabel)




load('C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/finalx_0.rda')

#filter n_xl for common genes
#n_xl[,1] is the gene entrezid
n_xl_coge_unlabel=list()
k=1

for(i in 1:length(n_xl_unlabel)){
  print(i)
  if (all(common_id%in%n_xl_unlabel[[i]][,1])){
    tmp=n_xl_unlabel[[i]][n_xl_unlabel[[i]][,1]%in%common_id,]
    n_xl_coge_unlabel[[k]]=tmp[order(tmp[,1]),-1]
    k=k+1
  }
  
  
}




dfx_unlabel=do.call(cbind,n_xl_coge_unlabel)#each row is a gene,each column is sample, rownames==common_id


undu_dfphe=dfphe_unlabel[!duplicated(dfphe_unlabel[,1]),]
if(all(!duplicated(dfphe_unlabel[,1]))==TRUE){
  print('noUdplicated unlabel')
}
save(dfx_unlabel,dfphe_unlabel,file='finalx_unlabel_0.rda')



###############################

dim(dfphe)
undu_dfphe=dfphe[!duplicated(dfphe[,1]),]
dim(undu_dfphe)

dim()
dim(dfx)
names(dfx_n_entrez)

names(n_xl_coge[[1]])

length(xlsam)

n_xl[[1]][,-"EntrezID"     ]

dfphe[,4]==tcolp
  tcol=colnames(dfx)
  colnames(dfx)[627]
  dfphe[627,4]
tcolp=gsub('[.]',' ',tcol )


?gsub
class(tt)
#undu_dfphe1=dfphe[!duplicated(dfphe[,4]),]
#dim(undu_dfphe1)
rownames( n_xl[[1]] )
names(xl[[1]])
class(xl[[1]])

pdi
!duplicated(samid)
pid


length(samid)
length(unique(samid))


for(i in 1:length(xl)){
  samid=c(samid,phel[[i]]['Sam_Name'][,1])
}

dim(xl[[1]])[1]

lf[43]
dim(xl[[43]])

samid=c()
for(i in 1:length(xl)){
  if(!is.null(xl[[i]])){
  if(dim(xl[[i]])[1]>30000){
    print(i)
    print(dim(xl[[i]]))
  }
  }
}
for(i in 1:length(n_xl)){
    print(dim(n_phel[[i]]))
    print(dim(n_xl_coge[[i]]))
}
length(xl[[2]])
length(phel)
length(xl)

table3=read.xlsx(lf[43],sheet=3)

length(xl)
xl[[43]]
length(samid)==length(unique(samid))

sum(duplicated(samid)!=duplicated(pid))
getcommongene<-function(){
  inter=c()
  for(i in length(xl)){
    temp=xl[-1,1]
    if(i==1){
      inter=temp
    }else{
      inter=intersect(c(inter,temp))
    }
  }
}


#######test xl
for(i in 1:55){
  print(paste0('start ',i))

  print( phel[[i]][,4]==gsub('[.]',' ',colnames(xl[[i]])[-1] ) )

}

phel[[41]][,4]
colnames(xl[[41]])[-1]

lf[41]
for(i in 1:55){
  print(paste0('start ',i))
  print('EntrezID'%in%colnames(xl[[i]]) )
}




phel[[53]][148,4]
colnames(xl[[53]])[-1][148]
library(bioMart)
library(biomaRt)
install.packages("BiocManager")
BiocManager::install("biomaRt", version = "3.8")

mart = useEnsembl("ENSEMBL_MART_ENSEMBL")
mart=useMart(biomart="ensembl", dataset="hsapiens_gene_ensembl")
bmIDs = getBM(attributes=c('ensembl_gene_id','ensembl_transcript_id',
                           'description',
                           'chromosome_name',
                           'start_position',
                           'end_position',
                           'strand','mgi_symbol','entrezgene'),mart = mart)


bmIDs = getBM(attributes=c('ensembl_gene_id','entrezgene'),mart = mart)

sum(complete.cases(bmIDs[,2]))


lungnet=read.table('C:/Users/Administrator/Desktop/deepgraphsurv/lung/lung_top/lung_top',fill=TRUE)
