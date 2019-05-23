setwd("C:/Users/Administrator/Downloads/TCGA_survival/TCGA-Assembler.2.0.6/TCGA-Assembler")#register email :xxz220@miami.edu
source("Module_A.R")
#for windows turnoff anti-virus
trace(utils:::unpackPkgZip, edit=TRUE)
#line 140 sleep(>1)

#breast
#lung
#prostate
#colon
# cancer type
#https://tcga-data.nci.nih.gov/docs/publications/tcga/?

filename_READ_RNASeq <- DownloadRNASeqData(cancerType = "BRCA",
                                           assayPlatform = "gene_RNAseq",
                                           tissueType = NULL,
                                           saveFolderName = "./BRCA_18_10_03",
                                           outputFileName = "brca.csv",
                                           inputPatientIDs = NULL)


filename_READ_RNASeq <- DownloadmiRNASeqData(cancerType = "BRCA",
                                             assayPlatform = "mir_HiSeq.hg19.mirbase20" ,
                                           tissueType = NULL,
                                           saveFolderName = "./BRCA_18_10_03",
                                           outputFileName = "brca_miRNA.csv",
                                           inputPatientIDs = NULL)

mi_a=read.csv("C:/Users/Administrator/Downloads/TCGA_survival/TCGA-Assembler.2.0.6/TCGA-Assembler/BRCA_18_10_03/brca_miRNA.txt",sep="\t",header=TRUE)

si=seq(2,dim(mi_a)[2],2)#this is necessary not use scaled_estimate,but use raw count for deseq
mi_a=mi_a[,c(1,si)]
mi_a=mi_a[-1,]
rownames(mi_a)<-mi_a[,1]
mi_a=mi_a[,-1]


dim(mi_a)
f_names=fetch_sample_id(names(mi_a),'[.]')
sur_names=fetch_sample_id(surcsv_r[,2],'-')
reserved_index=c()
age=c()
age_index=c()
for(i in c(1: ( length(f_names) ))){
  if(is.na(as.character(surcsv_r[which(f_names[i]==sur_names)[1],21])) ){
    next
  }
  if(as.character(surcsv_r[which(f_names[i]==sur_names)[1],21])!="--"){
    age_index=c(age_index,i)
    age=c(age,-as.numeric(as.character
                         (surcsv_r[which(f_names[i]==sur_names)[1],21])))
  }
}
mi_a=mi_a[,age_index]
dim(mi_a)
dim(surcsv_r)
names(surcsv_r[,21])

b=lapply(strsplit(x=names(mi_a),split="[.]"),function(x){

  bs=substr(x[4],1,1)

  if(bs==0){
    return("t")
  }else{
    return("n")
  }})
b=unlist(b)

#c is batch
c=lapply(strsplit(x=names(mi_a),split="[.]"),function(x){
  x[6]})
c=unlist(c)
age=as.numeric(age)

cold=cbind(as.factor(b),as.factor(c),age)
colnames(cold)=c("condition","batch","age")
dfcold=data.frame(cold)
dfcold$condition=factor(dfcold$condition)
dfcold$batch=factor(dfcold$batch)

mi_dma=data.matrix(mi_a)

mi_brcads <- DESeqDataSetFromMatrix(
  countData = mi_dma,colData=dfcold,design=~batch+condition+age)
mi_brcadeseq=DESeq(mi_brcads,test='LRT',reduced=~batch+age)
saveRDS(mi_brcadeseq,file="mi_brca.rds")
#clinical data
DownloadBiospecimenClinicalData(cancerType= "BRCA", saveFolderName = "./BiospecimenClinicalData", outputFileName = "")





#other cancers

filename_READ_RNASeq <- DownloadRNASeqData(cancerType = "LGG",
                                           assayPlatform = "gene_RNAseq",
                                           tissueType = NULL,
                                           saveFolderName = "./LGG_18_10_03",
                                           outputFileName = "LGG.csv",
                                           inputPatientIDs = NULL)

filename_READ_RNASeq <- DownloadRNASeqData(cancerType = "BLCA",
                                           assayPlatform = "gene_RNAseq",
                                           tissueType = NULL,
                                           saveFolderName = "./BLCA_18_10_03",
                                           outputFileName = "BLCA.csv",
                                           inputPatientIDs = NULL)
filename_READ_RNASeq <- DownloadRNASeqData(cancerType = "COAD",
                                           assayPlatform = "gene_RNAseq",
                                           tissueType = NULL,
                                           saveFolderName = "./COAD_18_10_03",
                                           outputFileName = "COAD.csv",
                                           inputPatientIDs = NULL)
filename_READ_RNASeq <- DownloadRNASeqData(cancerType = "GBM",
                                           assayPlatform = "gene_RNAseq",
                                           tissueType = NULL,
                                           saveFolderName = "./GBM_18_10_03",
                                           outputFileName = "GBM.csv",
                                           inputPatientIDs = NULL)
filename_READ_RNASeq <- DownloadRNASeqData(cancerType = "HNSC",
                                           assayPlatform = "gene_RNAseq",
                                           tissueType = NULL,
                                           saveFolderName = "./HNSC_18_10_03",
                                           outputFileName = "HNSC.csv",
                                           inputPatientIDs = NULL)
filename_READ_RNASeq <- DownloadRNASeqData(cancerType = "KIRC",
                                           assayPlatform = "gene_RNAseq",
                                           tissueType = NULL,
                                           saveFolderName = "./KIRC_18_10_03",
                                           outputFileName = "KIRC.csv",
                                           inputPatientIDs = NULL)
filename_READ_RNASeq <- DownloadRNASeqData(cancerType = "LUAD",
                                           assayPlatform = "gene_RNAseq",
                                           tissueType = NULL,
                                           saveFolderName = "./LUAD_18_10_03",
                                           outputFileName = "LUAD.csv",
                                           inputPatientIDs = NULL)
filename_READ_RNASeq <- DownloadRNASeqData(cancerType = "LGG",
                                           assayPlatform = "gene_RNAseq",
                                           tissueType = NULL,
                                           saveFolderName = "./LGG_18_10_03",
                                           outputFileName = "LGG.csv",
                                           inputPatientIDs = NULL)
filename_READ_RNASeq <- DownloadRNASeqData(cancerType = "LUSC",
                                           assayPlatform = "gene_RNAseq",
                                           tissueType = NULL,
                                           saveFolderName = "./LUSC_18_10_03",
                                           outputFileName = "LUSC.csv",
                                           inputPatientIDs = NULL)
filename_READ_RNASeq <- DownloadRNASeqData(cancerType = "OV",
                                           assayPlatform = "gene_RNAseq",
                                           tissueType = NULL,
                                           saveFolderName = "./OV_18_10_03",
                                           outputFileName = "OV.csv",
                                           inputPatientIDs = NULL)
filename_READ_RNASeq <- DownloadRNASeqData(cancerType = "PRAD",
                                           assayPlatform = "gene_RNAseq",
                                           tissueType = NULL,
                                           saveFolderName = "./PRAD_18_10_03",
                                           outputFileName = "PRAD.csv",
                                           inputPatientIDs = NULL)
filename_READ_RNASeq <- DownloadRNASeqData(cancerType = "SKCM",
                                           assayPlatform = "gene_RNAseq",
                                           tissueType = NULL,
                                           saveFolderName = "./SKCM_18_10_03",
                                           outputFileName = "SKCM.csv",
                                           inputPatientIDs = NULL)
filename_READ_RNASeq <- DownloadRNASeqData(cancerType = "STAD",
                                           assayPlatform = "gene_RNAseq",
                                           tissueType = NULL,
                                           saveFolderName = "./STAD_18_10_03",
                                           outputFileName = "STAD.csv",
                                           inputPatientIDs = NULL)
filename_READ_RNASeq <- DownloadRNASeqData(cancerType = "UCEC",
                                           assayPlatform = "gene_RNAseq",
                                           tissueType = NULL,
                                           saveFolderName = "./UCEC_18_10_03",
                                           outputFileName = "UCEC.csv",
                                           inputPatientIDs = NULL)
filename_READ_RNASeq <- DownloadRNASeqData(cancerType = "THCA",
                                           assayPlatform = "gene_RNAseq",
                                           tissueType = NULL,
                                           saveFolderName = "./THCA_18_10_03",
                                           outputFileName = "THCA.csv",
                                           inputPatientIDs = NULL)



#clinical data
DownloadBiospecimenClinicalData(cancerType= "LGG", saveFolderName = "./BiospecimenClinicalData", outputFileName = "")






#end other cancers




source("https://bioconductor.org/biocLite.R")
biocLite("DESeq2")
biocLite('GENIE3')
library(DESeq2)
library(GENIE3)
#browseVignettes("DESeq")
setwd("C:/Users/Administrator/Downloads/TCGA_survival/TCGA-Assembler.2.0.6/TCGA-Assembler")
a=read.csv("C:/Users/Administrator/Downloads/TCGA_survival/TCGA-Assembler.2.0.6/TCGA-Assembler/BRCA_18_10_03/BRCA.txt",sep="\t",header=TRUE)

names(a)
#this a is for obtaining raw count and TPM data. column with even number is raw count,...
dim(a)
si=seq(2,dim(a)[2],2)#this is necessary not use scaled_estimate,but use raw count for deseq
a=a[,c(1,si)]
a=a[-1,]
rownames(a)<-a[,1]
a=a[,-1]
#this is the final a ,buy the above 8 lines of code from a=read.csv(...)
a[1:5,1]




#b b is tumor("t") or normal("n")
#c is batch
#see https://wiki.nci.nih.gov/display/TCGA/TCGA+barcode   or search  TCGA barcode
b=lapply(strsplit(x=names(a),split="[.]"),function(x){

  bs=substr(x[4],1,1)

  if(bs==0){
    return("t")
  }else{
    return("n")
  }})
b=unlist(b)

#c is batch
c=lapply(strsplit(x=names(a),split="[.]"),function(x){
  x[6]})
c=unlist(c)
cold=cbind(b,c)
colnames(cold)=c("condition","batch")

dfcold=data.frame(cold)



dma=data.matrix(a)

source("https://bioconductor.org/biocLite.R")
biocLite("DESeq2")
library(DESeq2)

brcads <- DESeqDataSetFromMatrix(countData = dma,colData=dfcold,design=~batch+condition)

save(brcads,file='brca_frommatrix.RData')
saveRDS(brcads,file='brca_frommatrix.rds')
brcadeseq=DESeq(brcads)
saveRDS(brcadeseq,file='brca_deseq.rds')


brca_rds=readRDS(file='C:/Users/Administrator/Downloads/TCGA_survival/TCGA-Assembler.2.0.6/TCGA-Assembler/BRCA_18_10_03/BRCA_DESeq.RDS')

brca_lrt_rds=readRDS(file='C:/Users/Administrator/Downloads/TCGA_survival/TCGA-Assembler.2.0.6/TCGA-Assembler/BRCA_18_10_03/BRCADESeq_LRT.rds')

brca_mi=readRDS(file='C:/Users/Administrator/Downloads/TCGA_survival/TCGA-Assembler.2.0.6/TCGA-Assembler/BRCA_18_10_03/mi_brca.rds')



brca_result=DESeq2::results(brca_lrt_rds)
#brca_result=DESeq2::results(brca_rds,alpha=1-1e-7)
brca_result=DESeq2::results(brca_rds)
brca_result=DESeq2::results(brca_mi)

brca_result[which(rownames(brca_result))]

sa=brca_result[which(abs(brca_result$log2FoldChange)>1 & brca_result$padj<0.001),]

brca_result[which( brca_result$padj<0.05),]


#get real data matrix brca

#use the line below  on 3:33pm 2018-10-26
index1=which(abs(brca_result$log2FoldChange)>0 & brca_result$padj<0.05)
index1=which(abs(brca_result$log2FoldChange)>1 & brca_result$padj<0.05)
rnames=rownames(a[index1,])
rnames=rownames( brca_result[index1,] )
rnames_unfilter=rownames( brca_result )
c('TP53','PIK3CA','GATA3')%in%unlist(lapply(strsplit(rnames,"[|]"),function(x){x[1]}))

c('TP53','PIK3CA','GATA3')%in%unlist(lapply(strsplit(rnames_unfilter,"[|]"),function(x){x[1]}))


rownames( brca_result[index1,] )

brca_result[which(unlist(lapply(strsplit(rnames_unfilter,"[|]"),function(x){x[1]}))=='GATA3'),]

l2=unlist(lapply(strsplit(rnames_unfilter,"[|]"),function(x){x[1]}))

l2[3000:3300]

finded=unlist(lapply(strsplit(rnames,"[|]"),function(x){x[1]}))
'CPT1A'%in%unlist(lapply(strsplit(rnames,"[|]"),function(x){x[1]}))
c('CPT1A', 'CXCR7', 'GLA', 'HRASLS', 'NOTCH2NL', 'PGK1', 'PIK3CA', 'TTC3', 'UBXN7','ZFC3H1')%in%
unlist(lapply(strsplit(rnames,"[|]"),function(x){x[1]}))



library(glmnet)
##########survival analysis

surcsv=read.csv('C:/Users/Administrator/Downloads/TCGA_survival/survival_clinical/clinical.project-TCGA-BRCA.2018-10-03/clinical.tsv',
                sep='\t',header=T)


#tumor_a = a[,b=='t']



fetch_sample_id<-function(vec,symbol='[.]'){
  vec=as.character(vec)
return (
  unlist( lapply(strsplit(vec,symbol),function(x){x[[3]]}) )
)
}


surcsv_r = surcsv
rm1=c()
if (length(levels(surcsv_r[,which(names(surcsv_r)=='vital_status')]))==3){
  rm1=which(surcsv_r[,which(names(surcsv_r)=='vital_status')]=='--')
}

rm2=which((surcsv_r[,which(names(surcsv_r)=='vital_status')]=="dead"&
            surcsv_r[,which(names(surcsv_r)=='days_to_death')]=="--")|
            (surcsv_r[,which(names(surcsv_r)=='vital_status')]=="alive"&
surcsv_r[,which(names(surcsv_r)=='days_to_last_follow_up')]=="--"))



surcsv_r = surcsv_r[-c(union.Vector(rm1,rm2)),]

#still duplicate samples are not removed
clinical_samples=fetch_sample_id(surcsv_r[,2],'-')

rnames=rownames( brca_result[index1,] )

filtered_gene=unlist(lapply(strsplit(rnames,"[|]"),function(x){x[2]}))

gene_a=unlist(lapply(strsplit(rownames(a),"[|]"),function(x){x[2]}))

x_matrix=a[which(gene_a%in%filtered_gene ),which(b=='t'&(
  fetch_sample_id(names(a),'[.]')%in%clinical_samples ))]

x_matrix_col_names=fetch_sample_id(colnames(x_matrix),'[.]')



del_index=c()
for(i in c(1:(length(colnames(x_matrix))-1))){
  if (x_matrix_col_names[i] == x_matrix_col_names[i+1]){
    del_index=c(del_index,i+1)
  }
}
x_matrix=x_matrix[,-del_index]
#now duplicate samples are removed in columns of x_matrix

sorted_x=x_matrix[,order(fetch_sample_id(colnames(x_matrix)))]


surcsv_r=surcsv_r[fetch_sample_id(surcsv_r[,2],'-')%in%
                    fetch_sample_id(colnames(x_matrix)),]

surcsv_r=surcsv_r[order(fetch_sample_id(surcsv_r[,2],'-')),]
#1 death 0 alive(censored)
sur_event = as.numeric( surcsv_r[,which(names(surcsv_r)=='vital_status')]=='dead')
sur_time=rep(0,length(sur_event))
#obtain death days for uncensored data
sur_time[sur_event==1] = as.double( as.character(
  surcsv_r[sur_event==1,which(names(surcsv_r)=='days_to_death')] ) )
#obtain days to last_follow_up ,which is right censored samples
sur_time[sur_event==0] = as.double( as.character(
  surcsv_r[sur_event==0,which(names(surcsv_r)=='days_to_last_follow_up')] ) )

negsamples=which(sur_time<=0)

sorted_x=sorted_x[,-negsamples]
surcsv_r=surcsv_r[-negsamples,]

sur_event = as.numeric( surcsv_r[,which(names(surcsv_r)=='vital_status')]=='dead')
sur_time=rep(0,length(sur_event))
#obtain death days for uncensored data
sur_time[sur_event==1] = as.double( as.character(
  surcsv_r[sur_event==1,which(names(surcsv_r)=='days_to_death')] ) )
#obtain days to last_follow_up ,which is right censored samples
sur_time[sur_event==0] = as.double( as.character(
  surcsv_r[sur_event==0,which(names(surcsv_r)=='days_to_last_follow_up')] ) )


sur_z <- Surv(sur_time, sur_event)



cv.fit <- cv.glmnet(t(sorted_x),sur_z,family="cox",alpha=1)
plot(cv.fit)
fit_min<-glmnet(t(sorted_x),sur_z,family="cox",alpha=1,lambda = cv.fit$lambda.min)
fit_min$beta[which(fit_min$beta[,1]!=0),1]
length(names_a)
length(unique(names_a))
which(names_a=='A15A')
library(GENIE3)
library(DESeq2)
library(survival)
#preprocessing begin

a_addr="C:/Users/Administrator/Downloads/TCGA_survival/TCGA-Assembler.2.0.6/TCGA-Assembler/BLCA_18_10_03/BLCA.txt"
rds_addr='C:/Users/Administrator/Downloads/TCGA_survival/TCGA-Assembler.2.0.6/TCGA-Assembler/BLCA_18_10_03/BLCA_rdds.rds'
sur_addr='C:/Users/Administrator/Downloads/TCGA_survival/survival_clinical/clinical.project-TCGA-BLCA.2018-10-08/clinical.tsv'

a_addr="C:/Users/Administrator/Downloads/TCGA_survival/TCGA-Assembler.2.0.6/TCGA-Assembler/KIRC_18_10_03/KIRC.txt"
rds_addr='C:/Users/Administrator/Downloads/TCGA_survival/TCGA-Assembler.2.0.6/TCGA-Assembler/KIRC_18_10_03/KIRC_rdds.rds'
sur_addr='C:/Users/Administrator/Downloads/TCGA_survival/survival_clinical/clinical.project-TCGA-KIRC.2018-10-08/clinical.tsv'

rds_addr='C:/Users/Administrator/Downloads/TCGA_survival/TCGA-Assembler.2.0.6/TCGA-Assembler/BRCA_18_10_03/mi_brca.rds'



writedir="C:/Users/Administrator/Downloads/TCGA_survival/TCGA-Assembler.2.0.6/TCGA-Assembler/BLCA_18_10_03"

#BRCA
a_addr="C:/Users/Administrator/Downloads/TCGA_survival/TCGA-Assembler.2.0.6/TCGA-Assembler/BRCA_18_10_03/BRCA.txt"
rds_addr='C:/Users/Administrator/Downloads/TCGA_survival/TCGA-Assembler.2.0.6/TCGA-Assembler/BRCA_18_10_03/BRCA_DESeq.RDS'
sur_addr='C:/Users/Administrator/Downloads/TCGA_survival/survival_clinical/clinical.project-TCGA-BRCA.2018-10-03/clinical.tsv'


library(DESeq2)
library(GENIE3)
get_prep<-function(a_addr="C:/Users/Administrator/Downloads/TCGA_survival/TCGA-Assembler.2.0.6/TCGA-Assembler/BRCA_18_10_03/BRCA.txt"
                   ,rds_addr='C:/Users/Administrator/Downloads/TCGA_survival/TCGA-Assembler.2.0.6/TCGA-Assembler/BRCA_18_10_03/BRCA_DESeq.RDS'
                   ,sur_addr='C:/Users/Administrator/Downloads/TCGA_survival/survival_clinical/clinical.project-TCGA-BRCA.2018-10-03/clinical.tsv'
                   ,fd_change=1,p_adj=0.05,writedir="C:/Users/Administrator/Downloads/TCGA_survival/TCGA-Assembler.2.0.6/TCGA-Assembler/BRCA_18_10_03"){
  a=read.csv(a_addr,sep="\t",header=TRUE,stringsAsFactors = FALSE)

  #taa=read.table(a_addr,sep="\t",header=TRUE)
  si=seq(3,dim(a)[2],2)#for gcn ,we should use TPM
  a=a[,c(1,si)]
  a=a[-1,]
  rownames(a)<-a[,1]
  rownames_a=a[,1]
  a=a[,-1]

  #b b is tumor("t") or normal("n")
  #c is batch
  #see https://wiki.nci.nih.gov/display/TCGA/TCGA+barcode   or search  TCGA barcode
  b=lapply(strsplit(x=names(a),split="[.]"),function(x){

    bs=substr(x[4],1,1)

    if(bs==0){
      return("t")
    }else{
      return("n")
    }})
  b=unlist(b)

  a=apply(a,2,as.numeric)
  rownames(a)<-rownames_a
  #brca_rds=readRDS(file=rds_addr)
  #brca_result=DESeq2::results(brca_rds)
  v=apply(a,1,function(x){
    var(x)/( abs(mean(x))+5.015432e-298 )
  })

  index1=which(rank(v)>0.6*length(v))

  surcsv=read.csv(sur_addr,
                  sep='\t',header=T)


  #tumor_a = a[,b=='t']



  fetch_sample_id<-function(vec,symbol='[.]'){
    vec=as.character(vec)
    return (
      unlist( lapply(strsplit(vec,symbol),function(x){x[[3]]}) )
    )
  }


  surcsv_r = surcsv
  rm1=c()
  if (length(levels(surcsv_r[,which(names(surcsv_r)=='vital_status')]))==3){
    rm1=which(surcsv_r[,which(names(surcsv_r)=='vital_status')]=='--')
  }

  rm2=which((surcsv_r[,which(names(surcsv_r)=='vital_status')]=="dead"&
               surcsv_r[,which(names(surcsv_r)=='days_to_death')]=="--")|
              (surcsv_r[,which(names(surcsv_r)=='vital_status')]=="alive"&
                 surcsv_r[,which(names(surcsv_r)=='days_to_last_follow_up')]=="--"))



  surcsv_r = surcsv_r[-c(union.Vector(rm1,rm2)),]

  #still duplicate samples are not removed
  clinical_samples=fetch_sample_id(surcsv_r[,2],'-')

  rnames=rownames( a[index1,] )
  rnames_unfilter=rownames( a )
  #add in known important
  website=c('PIK3CA','TP53','GATA3','MAP3K1','MLL3','CDH1','NCOR1','MAP2K4','PTEN','RUNX1','PIK3R1','CTCF','AKT1','CBFB','SPEN',
            'SF3B1','ARID1A','RB1','MLL','KRAS','TBX3','ERBB2','FOXA1','MED23','STAG2','MYB','TBL1XR1','HIST1H3B','CASP8','CDKN1B','CUL4B',
            'RAB40A','ERBB3','CDC42BPA','SETDB1','FGFR2','GNPTAB','EP300','ACVR1B')

  rna_progn=c('FAM199X','GMCL1','CPT1A','FAM91A1','OTUD6B','ADAT1','ANKRD52','HRASLS','TRIM23','DAAM1','ME1','PIK3CA','GLA','TTC3','FRZB','PDSS2','UBR5','CXCR7','DIP2B','MCM10','ACSL1','HSP90AA1','NOTCH2NL','SMG1',
              'PTAR1','UBXN7','BIRC6','NDRG1','ZFC3H1','PGK1')

  important=union.Vector(website,rna_progn)

  extraindex=unlist(lapply(strsplit(rnames_unfilter,"[|]"),function(x){
    if (x[1]%in%important){
      return (x[2])
    }}))

  filtered_gene=unlist(lapply(strsplit(rnames,"[|]"),function(x){x[2]}))
  filtered_gene=union.Vector(filtered_gene,extraindex)
  gene_a=unlist(lapply(strsplit(rownames(a),"[|]"),function(x){x[2]}))

  x_matrix=a[which(gene_a%in%filtered_gene ),which(b=='t'&(
    fetch_sample_id(names(a),'[.]')%in%clinical_samples ))]

  x_matrix_col_names=fetch_sample_id(colnames(x_matrix),'[.]')



  del_index=c()
  for(i in c(1:(length(colnames(x_matrix))-1))){
    if (x_matrix_col_names[i] == x_matrix_col_names[i+1]){
      del_index=c(del_index,i+1)
    }
  }
  if (!is.null(del_index)){
    x_matrix=x_matrix[,-del_index]
  }
  #now duplicate samples are removed in columns of x_matrix

  sorted_x=x_matrix[,order(fetch_sample_id(colnames(x_matrix)))]


  surcsv_r=surcsv_r[fetch_sample_id(surcsv_r[,2],'-')%in%
                      fetch_sample_id(colnames(x_matrix)),]

  surcsv_r=surcsv_r[order(fetch_sample_id(surcsv_r[,2],'-')),]
  #1 death 0 alive(censored)
  sur_event = as.numeric( surcsv_r[,which(names(surcsv_r)=='vital_status')]=='dead')
  sur_time=rep(0,length(sur_event))
  #obtain death days for uncensored data
  sur_time[sur_event==1] = as.double( as.character(
    surcsv_r[sur_event==1,which(names(surcsv_r)=='days_to_death')] ) )
  #obtain days to last_follow_up ,which is right censored samples
  sur_time[sur_event==0] = as.double( as.character(
    surcsv_r[sur_event==0,which(names(surcsv_r)=='days_to_last_follow_up')] ) )

  negsamples=which(sur_time<=0)

  sorted_x=sorted_x[,-negsamples]
  surcsv_r=surcsv_r[-negsamples,]

  sur_event = as.numeric( surcsv_r[,which(names(surcsv_r)=='vital_status')]=='dead')
  sur_time=rep(0,length(sur_event))
  #obtain death days for uncensored data
  sur_time[sur_event==1] = as.double( as.character(
    surcsv_r[sur_event==1,which(names(surcsv_r)=='days_to_death')] ) )
  #obtain days to last_follow_up ,which is right censored samples
  sur_time[sur_event==0] = as.double( as.character(
    surcsv_r[sur_event==0,which(names(surcsv_r)=='days_to_last_follow_up')] ) )


  #sur_z <- Surv(sur_time, sur_event)
  #GENIE3 requires that the row names be gene and column names be samples
  weightMatrix=GENIE3(as.matrix(sorted_x))
  setwd(writedir)
  write.csv(t(sorted_x),file='x1.csv')
  write.csv(sur_time,file='t.csv')
  write.csv(sur_event,file='c.csv')
  write.csv(weightMatrix,file='w.csv')
  #return (list(x=t(sorted_x),z=sur_z,z_time=sur_time,censored=sur_event,wm=weightMatrix))

}

setwd("C:/Users/Administrator/Downloads/TCGA_survival/TCGA-Assembler.2.0.6/TCGA-Assembler/BRCA_18_10_03")

write.csv(t(sorted_x),file='x.csv')
write.csv(sur_time,file='t.csv')
write.csv(sur_event,file='c.csv')
write.csv(weightMatrix,file='w.csv')
#preprocessing end


######GENIE3
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("GENIE3", version = "3.8")


#start recycle bin
rnames_unfilter=rownames( brca_result )

brca_result[index1,]

c('TP53','PIK3CA','GATA3')%in%unlist(lapply(strsplit(rnames,"[|]"),function(x){x[1]}))

c('TP53','PIK3CA','GATA3')%in%unlist(lapply(strsplit(rnames_unfilter,"[|]"),function(x){x[1]}))

brca_result[which(unlist(lapply(strsplit(rnames_unfilter,"[|]"),function(x){x[1]}))=='PIK3CA'),]

brca_result[which(unlist(lapply(strsplit(rnames_unfilter,"[|]"),function(x){x[1]}))=='BCL2'),]
l2[which(unlist(lapply(strsplit(rnames_unfilter,"[|]"),function(x){x[1]}))=='TP53')]


#end recycle bin


source("https://bioconductor.org/biocLite.R")
biocLite('GENIE3')
library(GENIE3)
browseVignettes("GENIE3")







###############test download

filename_READ_RNASeq <- DownloadRNASeqData(cancerType = "BRCA",
                                           assayPlatform = "exon_RNAseq",
                                           tissueType = NULL,
                                           saveFolderName = "./BRCA_19_02_03",
                                           outputFileName = "brca.csv",
                                           inputPatientIDs = NULL)



