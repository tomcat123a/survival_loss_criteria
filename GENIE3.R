#library(DESeq2)
library(GENIE3)
args = commandArgs(trailingOnly=TRUE)#arg[1] is the first parameter
get_prep<-function(a_addr=paste0("/scratch/tmp/pudge/gex/src/",args[1],'.txt')
                   ,sur_addr=paste0('/scratch/tmp/pudge/gex/surv/clinical.project-TCGA-',args[1],'.2018-10-08/clinical.tsv')
                   ,writedir=paste0("/scratch/tmp/pudge/gex/result/")){
  a=read.csv(a_addr,sep="\t",header=TRUE,stringsAsFactors = FALSE)
  
  print(args[1])
  si=seq(3,dim(a)[2],2)#this is necessary not use scaled_estimate,but use raw count for deseq
  a=a[,c(1,si)]
  a=a[-1,]
  rownames(a)<-a[,1]
  rownames_a=a[,1]
  
  a=a[,-1]
  colnames_a=names(a)
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
  colnames(a)<-colnames_a
  #brca_rds=readRDS(file=rds_addr)
  #brca_result=DESeq2::results(brca_rds)
  v1=apply(a,1,function(x){
      sum(x!=0)/length(x)
  })
  v2=apply(a,1,function(x){
      var(x) /(abs(mean(x))+1e-298) })
  index1=which(v1>0.95 & rank(v2)>0.7*length(v2))
   
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
  
  
  
  if (length(rm1)+length(rm2)>0){surcsv_r = surcsv_r[-c(union(rm1,rm2)),]}
  
  #still duplicate samples are not removed
  clinical_samples=fetch_sample_id(surcsv_r[,2],'-')
  
  rnames=rownames( a[index1,] )
  rnames_unfilter=rownames( a )
  
  filtered_gene=unlist(lapply(strsplit(rnames,"[|]"),function(x){x[2]}))
  if(args[1]=='BRCA'){
    #add in known important,only for BRCA
    website=c('PIK3CA','TP53','GATA3','MAP3K1','MLL3','CDH1','NCOR1','MAP2K4','PTEN','RUNX1','PIK3R1','CTCF','AKT1','CBFB','SPEN',
              'SF3B1','ARID1A','RB1','MLL','KRAS','TBX3','ERBB2','FOXA1','MED23','STAG2','MYB','TBL1XR1','HIST1H3B','CASP8','CDKN1B','CUL4B',
              'RAB40A','ERBB3','CDC42BPA','SETDB1','FGFR2','GNPTAB','EP300','ACVR1B')
    
    rna_progn=c('FAM199X','GMCL1','CPT1A','FAM91A1','OTUD6B','ADAT1','ANKRD52','HRASLS','TRIM23','DAAM1','ME1','PIK3CA','GLA','TTC3','FRZB','PDSS2','UBR5','CXCR7','DIP2B','MCM10','ACSL1','HSP90AA1','NOTCH2NL','SMG1',
                'PTAR1','UBXN7','BIRC6','NDRG1','ZFC3H1','PGK1')
    important=union(website,rna_progn)
    extraindex=unlist(lapply(strsplit(rnames_unfilter,"[|]"),function(x){
      if (x[1]%in%important){
        return (x[2])
      }}))
    filtered_gene=union(filtered_gene,extraindex)
  }
  
  
  
  #filtered_gene=unlist(lapply(strsplit(rnames,"[|]"),function(x){x[2]}))
  
  
  gene_a=unlist(lapply(strsplit(rownames(a),"[|]"),function(x){x[2]}))
  
  x_matrix=a[which(gene_a%in%filtered_gene ),which(b=='t'&(
    fetch_sample_id(colnames(a),'[.]')%in%clinical_samples ))]
  
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
  #weightMatrix=GENIE3(as.matrix(sorted_x))
  setwd(writedir)
  write.csv(t(sorted_x),file=paste0(args[1],'_x.csv'))
  write.csv(sur_time,file=paste0(args[1],'_t.csv'))
  write.csv(sur_event,file=paste0(args[1],'_c.csv'))
  #write.csv(weightMatrix,file=paste0(args[1],'_w.csv'))
  #return (list(x=t(sorted_x),z=sur_z,z_time=sur_time,censored=sur_event,wm=weightMatrix))
  
}

get_prep()
