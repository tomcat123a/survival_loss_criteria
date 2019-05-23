options(stringsAsFactors = FALSE)

library(data.table)
namelist=c('STAD','COAD','UCEC','THCA','KIRC','PRAD','BLCA','LUAD','LUSC','SKCM')
write_tcga<-function(cancer='BLCA'){
#for TCGA


  dfx=read.csv(paste0('C:/python_study/surviv/survivdeep/',cancer,'_x.csv'))#get dfx,dfphe
  #convert to entrezid
  for(i in 2:dim(dfx)[2]){
    colnames(dfx)[i]=tail(strsplit(colnames(dfx)[i],'[.]')[[1]],1)
  }
  tdfx=t(dfx)
  colnames(tdfx)=tdfx[1,]
  tdfx=tdfx[-1,]
  filtered_dfx = tdfx
  
  if(cancer=='COAD'){
    dt=fread('C:/python_study/surviv/humanbase/colon_top/colon_top',fill=TRUE)
  }
  if(cancer=='KIRC'){
    dt=fread('C:/python_study/surviv/humanbase/kidney_top/kidney_top',fill=TRUE)
  }
  if(cancer=='PRAD'){
    dt=fread('C:/python_study/surviv/humanbase/prostate_gland_top/prostate_gland_top',fill=TRUE)
  }
  if(cancer=='SKCM'){
    dt=fread('C:/python_study/surviv/humanbase/skin_top/skin_top',fill=TRUE)
  }
   
  if(cancer=='STAD'){
    dt=fread('C:/python_study/surviv/humanbase/stomach_top/stomach_top',fill=TRUE)
  }
  if(cancer=='THCA'){
    dt=fread('C:/python_study/surviv/humanbase/thyroid_gland_top/thyroid_gland_top',fill=TRUE)
  }
  if(cancer=='UCEC'){
    dt=fread('C:/python_study/surviv/humanbase/UCEC_top/UCEC_top',fill=TRUE)
  }
  if(cancer=='BLCA'){
    dt=fread('C:/python_study/surviv/humanbase/urinary_bladder_top/urinary_bladder_top',fill=TRUE)
  }
   if(cancer=='LUSC'){
     dt=fread('C:/Users/Administrator/Desktop/deepgraphsurv/lung/lung_top/lung_top',fill=TRUE)
   } 
  if(cancer=='LUAD'){
    dt=fread('C:/Users/Administrator/Desktop/deepgraphsurv/lung/lung_top/lung_top',fill=TRUE)
  }
  setwd("C:/Users/Administrator/Desktop/deepgraphsurv/rnaseq_cancer/survdata")
   
  
  
  #setwd("C:/Users/Administrator/Desktop/deepgraphsurv/lung_cancer/survdata")
  
   i=0
    
    write.csv(filtered_dfx,paste0(cancer,'_x_',i,'.csv'))
    
    #write.csv(dfphe[,2],paste0(cancer,'_t_',i,'.csv'))
    
    #write.csv(dfphe[,3],paste0(cancer,'_c_',i,'.csv'))
    
    #write.csv(dfx_unlabel[index1,],paste0(cancer,'_un_',i,'.csv'))
    
    name0=rownames(filtered_dfx)
    n_name0=as.numeric(name0)
    dt1=dt[dt$V1 %in% n_name0,]
    dt2=dt1[dt1$V2 %in% n_name0,]
    write.csv(dt2,file=paste0(cancer,'_network_',i,'.csv'))
}
for(i in 8:10){
  write_tcga(namelist[i])
}
