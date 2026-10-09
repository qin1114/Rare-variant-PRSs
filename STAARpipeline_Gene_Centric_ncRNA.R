# R --slave --no-restore --file=STAARpipeline_Gene_Centric_ncRNA.R --args 100 0.01 ../../data/STAARpipeline/ ../../results/STAARpipeline
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

source("./Burden_scores.R")


args <- commandArgs(trailingOnly = TRUE)
arrayid <- as.integer(args[1])
rare_maf_cutoff <- as.numeric(args[2])
gds_file <- as.character(args[3]) 
out_dir <- as.character(args[4]) 

cat("arrayid:", arrayid, "\n")
cat("rare_maf_cutoff:", rare_maf_cutoff, "\n")
cat("gds_file:", gds_file, "\n")
cat("out_dir:", out_dir, "\n")


## QC_label
QC_label <- "annotation/info/QC_label"
## variant_type
variant_type <- "variant"  # variant, SNV
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
## gene number in job
gene_num_in_array <- 100  # 100 
group.num.allchr <- ceiling(table(ncRNA_gene[,1])/gene_num_in_array)
sum(group.num.allchr)


chr <- which.max(arrayid <= cumsum(group.num.allchr))
group.num <- group.num.allchr[chr]

if (chr == 1){
  groupid <- arrayid
}else{
  groupid <- arrayid - cumsum(group.num.allchr)[chr-1]
}

ncRNA_gene_chr <- ncRNA_gene[ncRNA_gene[,1]==chr,]
sub_seq_num <- dim(ncRNA_gene_chr)[1]

if(groupid < group.num)
{
  sub_seq_id <- ((groupid - 1)*gene_num_in_array + 1):(groupid*gene_num_in_array)
}else
{
  sub_seq_id <- ((groupid - 1)*gene_num_in_array + 1):sub_seq_num
}


## exclude large ncRNA masks
if(arrayid==117)
{
  sub_seq_id <- setdiff(sub_seq_id,53)
}

if(arrayid==218)
{
  sub_seq_id <- setdiff(sub_seq_id,19)
}

if(arrayid==220)
{
  sub_seq_id <- setdiff(sub_seq_id,c(208,274))
}

if(arrayid==221)
{
  sub_seq_id <- setdiff(sub_seq_id,311)
}

if(arrayid==156)
{
  sub_seq_id <- setdiff(sub_seq_id,41)
}

if(arrayid==219)
{
  sub_seq_id <- setdiff(sub_seq_id,103)
}


## log_dir
dir.create(paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Noncoding/chr", chr), recursive = TRUE)
log_file=paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Noncoding/chr", chr, "/chr", chr, "_", variant_type, "_ncRNA.log")
cat(paste("Deal with chr", chr, "for array id", arrayid, "with", length(sub_seq_id), "genes!\n"), file=log_file)


## aGDS file
genofile <- seqOpen(paste0(gds_file, "/Q0_unre_Caucasian_chr", chr, ".gds"))
sampleids <- seqGetData(genofile,"sample.id")  # 345967


Sig_Burdens <- NULL
Rare_Gene_Columns <- NULL
ignore_gene <- NULL
execution_time <- system.time({
  for(kk in sub_seq_id)
  {
    # print(kk)
    gene_name <- ncRNA_gene_chr[kk,2]
    # results <- c()
    a <- Burden_Scores(region = 'Noncoding',chr = chr, gene_name = gene_name,category = category,
                    genofile = genofile,IDs = sampleids,rare_maf_cutoff=rare_maf_cutoff,rv_num_cutoff=2,
                    QC_label=QC_label,variant_type=variant_type,geno_missing_imputation=geno_missing_imputation,
                    Annotation_dir=Annotation_dir,Annotation_name_catalog=Annotation_name_catalog,silent = TRUE, log_file=log_file) 
    if(length(a) == 1){
        ignore_gene <- rbind(ignore_gene, data.frame(chr = chr, gene_name = gene_name, category=category))
    }else{  
        Sig_Burdens <- cbind(Sig_Burdens,a)
        Rare_Gene_Columns <- cbind(Rare_Gene_Columns, paste0(gene_name, "_", category))
    }
  }
})

cat(paste("Total execution time:", execution_time["elapsed"], "seconds\n\n"), file=log_file, append = TRUE)
seqClose(genofile)

rownames(Sig_Burdens) = sampleids
colnames(Sig_Burdens) <- Rare_Gene_Columns
ignore_gene = ignore_gene[ignore_gene$chr==chr,]
save(Sig_Burdens, file = paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/", coding,  "/chr", chr, "/", category, "/chr_", chr, "_", variant_type, "_", category, "_arrayid", arrayid, ".RData"))
save(ignore_gene, file = paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/", coding, "/chr", chr, "/", category, "/ignore_rare_gene_chr", chr, "_", variant_type, "_", category, "_arrayid", arrayid, ".RData"))  ## 每根染色体保存一次中间结果

