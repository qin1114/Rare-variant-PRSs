# R --slave --no-restore --file=STAARpipeline_Gene_Centric_Noncoding_Long_Masks.R --args 0.01 ../../data/STAARpipeline/ ../../results/STAARpipeline
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


## gene number in job
gene_num_in_array <- 50 
group.num.allchr <- ceiling(table(genes_info[,2])/gene_num_in_array)
sum(group.num.allchr)

ignore_gene <- data.frame(chr = character(), gene_name = character(), category=character(), stringsAsFactors = FALSE)
## analyze large noncoding masks
arrayid <- c(21,39,44,45,46,53,55,83,88,103,114,127,135,150,154,155,163,164,166,180,189,195,200,233,280,285,295,313,318,319,324,327,363,44,45,54)
sub_seq_id <- c(1009,1929,182,214,270,626,741,894,83,51,611,385,771,493,671,702,238,297,388,352,13,303,600,170,554,207,724,755,1048,319,324,44,411,195,236,677)

region_spec <- data.frame(arrayid,sub_seq_id) 
sub_seq_id <- 1:length(arrayid)

genes <- genes_info

## log_dir
dir.create(paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Noncoding/LongMask"), recursive = TRUE)
log_file=paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Noncoding/LongMask/", variant_type, "_longmask.log")
cat(paste("Deal with long mask", "with", length(sub_seq_id), "genes!\n"), file=log_file)


### Significant Rare Variant Set's Burden Scores
Sig_Burdens <- NULL
Rare_Gene_Columns <- NULL
execution_time <- system.time({   
    # Sig_Burdens <- NULL
    ### Loop over significant rare variant sets
    ### Using IDs = sampleids to build burden scores for all individuals.
    for(kk in sub_seq_id){
        print(kk)
        arrayid <- region_spec$arrayid[kk]
        sub_id <- region_spec$sub_seq_id[kk]
        chr <- which.max(arrayid <= cumsum(group.num.allchr))
        genes_info_chr <- genes_info[genes_info[,2]==chr,]
        gene_name <- genes_info_chr[sub_id,1]


        genofile <- seqOpen(paste0(gds_file, "/Q0_unre_Caucasian_chr", chr, ".gds"))
        sampleids <- seqGetData(genofile,"sample.id")  

        for(category in c("downstream","upstream","UTR","promoter_CAGE","promoter_DHS","enhancer_CAGE","enhancer_DHS")){
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
        cat(paste0(gene_name, " ", chr))
        seqClose(genofile)
    }
})

cat(paste("Total execution time:", execution_time["elapsed"], "seconds\n\n"), file=log_file, append = TRUE)

colnames(Sig_Burdens) <- Rare_Gene_Columns
rownames(Sig_Burdens) = sampleids
save(Sig_Burdens, file = paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Noncoding/LongMask/", variant_type, "_longmask.RData"))
save(ignore_gene, file = paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Noncoding/LongMask/ignore_rare_gene_longmask.RData")) 

