# R --slave --no-restore --file=STAARpipeline_Gene_Centric_Noncoding.R --args 100 0.01 ../../data/STAARpipeline/ ../../results/STAARpipeline
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


set.seed(1330)


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
variant_type <- "variant"  # variant
## geno_missing_imputation
geno_missing_imputation <- "mean"
## Annotation_dir
Annotation_dir <- "annotation/info/FunctionalAnnotation"
## Annotation channel
Annotation_name_catalog <- read.csv(paste0(out_dir, "/Step_0/Annotation_name_catalog.csv"))


## gene number in job
ignore_gene <- data.frame(chr = character(), gene_name = character(), category=character(), stringsAsFactors = FALSE)
gene_num_in_array <- 50  # 50
group.num.allchr <- ceiling(table(genes_info[,2])/gene_num_in_array)
sum(group.num.allchr)

chr <- which.max(arrayid <= cumsum(group.num.allchr))
group.num <- group.num.allchr[chr]

if (chr == 1){
  groupid <- arrayid
}else{
  groupid <- arrayid - cumsum(group.num.allchr)[chr-1]
}

genes_info_chr <- genes_info[genes_info[,2]==chr,]
sub_seq_num <- dim(genes_info_chr)[1]

if(groupid < group.num)
{
  sub_seq_id <- ((groupid - 1)*gene_num_in_array + 1):(groupid*gene_num_in_array)
}else
{
  sub_seq_id <- ((groupid - 1)*gene_num_in_array + 1):sub_seq_num
}

## exclude large noncoding masks
jobid_exclude <- c(21,39,44,45,46,53,55,83,88,103,114,127,135,150,154,155,163,164,166,180,189,195,200,233,280,285,295,313,318,319,324,327,363,44,45,54)
sub_seq_id_exclude <- c(1009,1929,182,214,270,626,741,894,83,51,611,385,771,493,671,702,238,297,388,352,13,303,600,170,554,207,724,755,1048,319,324,44,411,195,236,677)

for(i in 1:length(jobid_exclude))
{
  if(arrayid==jobid_exclude[i])
  {
    sub_seq_id <- setdiff(sub_seq_id,sub_seq_id_exclude[i])
  }
}



## log_dir
dir.create(paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Noncoding/chr", chr), recursive = TRUE)
log_file=paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Noncoding/chr", chr, "/arrayid", arrayid, "_", variant_type, ".log")
cat(paste("Deal with chr", chr, "for array id", arrayid, "with", length(sub_seq_id), "genes!\n"), file=log_file, append = TRUE)

### Significant Rare Variant Set's Burden Scores
genofile <- seqOpen(paste0(gds_file, "/Q0_unre_Caucasian_chr", chr, ".gds"))
sampleids <- seqGetData(genofile,"sample.id")  


Sig_Burdens <- NULL
Rare_Gene_Columns <- NULL
ignore_gene <- NULL
execution_time <- system.time({   
    # Sig_Burdens <- NULL
    ### Loop over significant rare variant sets
    ### Using IDs = sampleids to build burden scores for all individuals.
    for(kk in sub_seq_id){
        print(kk)
        gene_name <- genes_info_chr[kk,1]
        
        for(category in c("downstream","upstream","UTR","promoter_CAGE","promoter_DHS","enhancer_CAGE","enhancer_DHS")){
            a <- Burden_Scores(region = 'Noncoding',chr = chr, gene_name = gene_name,category = category,
                        genofile = genofile,IDs = sampleids,rare_maf_cutoff=rare_maf_cutoff,rv_num_cutoff=2,
                        QC_label=QC_label,variant_type=variant_type,geno_missing_imputation=geno_missing_imputation,
                        Annotation_dir=Annotation_dir,Annotation_name_catalog=Annotation_name_catalog,silent = TRUE, log_file=log_file) 

            if(length(a) == 1){  # The gene exists, but this type of mutation does not exist: set to 0.
                ignore_gene <- rbind(ignore_gene, data.frame(chr = chr, gene_name = gene_name, category=category))
                }else{  
                Sig_Burdens <- cbind(Sig_Burdens,a)
                Rare_Gene_Columns <- cbind(Rare_Gene_Columns, paste0(gene_name, "_", category))
            }
        }
    }
})

cat(paste("Total execution time:", execution_time["elapsed"], "seconds\n\n"), file=log_file, append = TRUE)
seqClose(genofile)

colnames(Sig_Burdens) <- Rare_Gene_Columns
rownames(Sig_Burdens) = sampleids

save(Sig_Burdens, file = paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Noncoding/chr", chr, "/chr_", chr, "_allrare_", variant_type, "_arrayid", arrayid, ".RData"))
save(ignore_gene, file = paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Noncoding/ignore_rare_gene_chr", chr, "_arrayid", arrayid, ".RData")) 


