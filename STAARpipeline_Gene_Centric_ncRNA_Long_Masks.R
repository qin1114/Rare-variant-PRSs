# R --slave --no-restore --file=STAARpipeline_Gene_Centric_ncRNA_Long_Masks.R --args 0.01 ../../data/STAARpipeline/ ../../results/STAARpipeline
library(gdsfmt) 
library(SeqArray) 
library(SeqVarTools) 
library(dplyr)
library(STAAR)  
library(GENESIS)
library(TxDb.Hsapiens.UCSC.hg38.knownGene) 
library(Matrix)  
library(SCANG)
library(STAARpipeline)  
library(glmnet)
library(caret)
library(SuperLearner)

library(R.utils)

source("./Burden_scores.R")


args <- commandArgs(trailingOnly = TRUE)
rare_maf_cutoff <- as.numeric(args[1])    
gds_file <- as.character(args[2]) 
out_dir <- as.character(args[3]) 

cat("rare_maf_cutoff:", rare_maf_cutoff, "\n")
cat("gds_file:", gds_file, "\n")
cat("out_dir:", out_dir, "\n")


## QC_label
QC_label <- "annotation/info/QC_label"
## variant_type
variant_type <- "variant"  # variant
## geno_missing_imputation
geno_missing_imputation <- "mean"
## Annotation_dir
Annotation_dir <- "annotation/info/FunctionalAnnotation"
## Annotation channel
Annotation_name_catalog <- read.csv(paste0(out_dir, "/Step_0/Annotation_name_catalog.csv"))


###########################################################
#           Main Function 
###########################################################
category = "ncRNA"
## analyze large ncRNA masks
arrayid <- c(117,218,220,220,221,156,219)
sub_seq_id <- c(53,19,208,274,311,41,103)

region_spec <- data.frame(arrayid,sub_seq_id)

## gene number in job
gene_num_in_array <- 100  # 100 
group.num.allchr <- ceiling(table(ncRNA_gene[,1])/gene_num_in_array)
sum(group.num.allchr)



## log_dir
log_file=paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Noncoding/LongMask/", variant_type, "_", category, "_long_mask.log")
cat(paste("Deal with long mask for", category, "with", length(sub_seq_id), "genes!\n"), file=log_file, append = TRUE) 


ignore_gene <- NULL
execution_time <- system.time({
  for(kk in 1:dim(region_spec)[1])
  {
    # print(kk)
    arrayid <- region_spec$arrayid[kk]
    sub_seq_id <- region_spec$sub_seq_id[kk]

    chr <- which.max(arrayid <= cumsum(group.num.allchr))
    ncRNA_gene_chr <- ncRNA_gene[ncRNA_gene[,1]==chr,] 

    ## aGDS file
    genofile <- seqOpen(paste0(gds_file, "/Q0_unre_Caucasian_chr", chr, ".gds"))
    sampleids <- seqGetData(genofile,"sample.id")  

    gene_name <- ncRNA_gene_chr[sub_seq_id,2]
    a <- Burden_Scores(region = 'Noncoding',chr = chr, gene_name = gene_name,category = category,
                    genofile = genofile,IDs = sampleids,rare_maf_cutoff=0.01,rv_num_cutoff=2,
                    QC_label=QC_label,variant_type=variant_type,geno_missing_imputation=geno_missing_imputation,
                    Annotation_dir=Annotation_dir,Annotation_name_catalog=Annotation_name_catalog,silent = TRUE, log_file=log_file) 

    if(length(a) == 1){ 
        ignore_gene <- rbind(ignore_gene, data.frame(chr = chr, gene_name = gene_name, category=category))
        save(ignore_gene, file = paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Noncoding/LongMask/ignore_rare_gene_", variant_type, "_", category, "_long_mask", ".RData"))
    }else{  
        Sig_Burdens <- a
        rownames(Sig_Burdens) = sampleids
        colnames(Sig_Burdens) <- paste0(gene_name, "_", category)
        save(Sig_Burdens, file = paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Noncoding/LongMask/", variant_type, "_", category, "_id", kk+1, "_long_mask", ".RData"))
        rm(a)
        rm(Sig_Burdens)
    }
    seqClose(genofile)
  }
})

cat(paste("Total execution time:", execution_time["elapsed"], "seconds\n\n"), file=log_file, append = TRUE)
save(ignore_gene, file = paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Noncoding/LongMask/ignore_rare_gene_", variant_type, "_", category, "_long_mask", ".RData"))  


