# R --slave --no-restore --file=merge_noncoding_burden.R --args 0.01 ../../results/STAARpipeline 
library(reticulate)
np <- import("numpy")

config <- py_config()
config$numpy
print(config)


args <- commandArgs(trailingOnly = TRUE)
rare_maf_cutoff <- as.numeric(args[1])
out_dir <- as.character(args[2]) 

cat("rare_maf_cutoff:", rare_maf_cutoff, "\n")
cat("out_dir:", out_dir, "\n")

save_path = paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Noncoding")
dir.create(paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Noncoding/Merge"), recursive = TRUE)
dir.create(paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Noncoding/NPY_FILE"), recursive = TRUE)


## Noncoding
## gene number in job
genes_info = get(load(out_dir, "/Step_0/genes_info_STAAR.RData")) # load from STAAR library
gene_num_in_array <- 50  
group.num.allchr <- ceiling(table(genes_info[,2])/gene_num_in_array)

job_cum = 1
for(chr in 1:22){
    job_num_chr = group.num.allchr[chr]
    Noncoding_file = paste0(save_path, "/chr", chr)
    Noncoding_chr = NULL
    ignore_chr = NULL
    for(array_id in job_cum:(job_cum+job_num_chr-1)){
        if (file.exists(paste0(Noncoding_file, "/chr_", chr,"_allrare_variant_arrayid", array_id, ".RData"))){
            Noncoding_chr_cur = get(load(paste0(Noncoding_file, "/chr_", chr,"_allrare_variant_arrayid", array_id, ".RData")))
            Noncoding_chr = cbind(Noncoding_chr, Noncoding_chr_cur)
            rm(Noncoding_chr_cur)
            ignore_chr_cur = get(load(paste0(save_path, "/ignore_rare_gene_chr", chr,"_arrayid", array_id, ".RData")))
            ignore_chr = rbind(ignore_chr, ignore_chr_cur)
            rm(ignore_chr_cur)
        }
        
    }
    print(paste("deal with array id from", job_cum, "to", (job_cum+job_num_chr-1), "for chr", chr))
    job_cum = job_cum + job_num_chr
    save(Noncoding_chr, file=paste0(save_path, "/Merge/chr", chr, "_variant_Noncoding_merged.RData"))
    save(ignore_chr, file=paste0(save_path, "/Merge/ignore_rare_gene_chr", chr, "_variant_Noncoding_merged.RData"))

    data_array <- as.matrix(Noncoding_chr)
    np$savez(paste0(save_path, "/NPY_FILE/chr", chr, "_variant_Noncoding_merged.npz"),
         data = data_array,
         colnames = colnames(Noncoding_chr),
         sampleid = rownames(Noncoding_chr))

    rm(Noncoding_chr)
    rm(ignore_chr)
}



# Noncoding longmask
rdata_coding <- get(load(paste0(save_path, "/LongMask/Noncoding_allrare_variant_longmask.RData")))
data_array <- as.matrix(rdata_coding)

np$savez(paste0(save_path, "/NPY_FILE/Noncoding_allrare_variant_longmask.npz"),
        data = data_array,
        colnames = colnames(rdata_coding),
        sampleid = rownames(rdata_coding))



## ncRNA
## gene number in job
ncRNA_gene = get(load(out_dir, "/Step_0/ncRNA_gene_STAAR.RData"))
gene_num_in_array <- 100  # 100 
group.num.allchr <- ceiling(table(ncRNA_gene[,1])/gene_num_in_array)

job_cum = 1
ncRNA_all = NULL
ignore_all = NULL
for(chr in 1:22){
    job_num_chr = group.num.allchr[chr]
    ncRNA_file = paste0(save_path, "/chr", chr, "/ncRNA/")
    ncRNA_chr = NULL
    ignore_chr = NULL
    for(array_id in job_cum:(job_cum+job_num_chr-1)){
        if (file.exists(paste0(ncRNA_file, "chr_", chr,"_variant_ncRNA_arrayid", array_id, ".RData"))){
            ncRNA_chr_cur = get(load(paste0(ncRNA_file, "chr_", chr,"_variant_ncRNA_arrayid", array_id, ".RData")))
            ncRNA_chr = cbind(ncRNA_chr, ncRNA_chr_cur)
            rm(ncRNA_chr_cur)
            ignore_chr_cur = get(load(paste0(ncRNA_file, "ignore_rare_gene_chr", chr,"_variant_ncRNA_arrayid", array_id, ".RData")))
            ignore_chr = rbind(ignore_chr, ignore_chr_cur)
            rm(ignore_chr_cur)
        }
        
    }
    print(paste("deal with array id from", job_cum, "to", (job_cum+job_num_chr-1), "for chr", chr))
    job_cum = job_cum + job_num_chr
    ncRNA_all = cbind(ncRNA_all, ncRNA_chr)
    ignore_all = rbind(ignore_all, ignore_chr)
    rm(ncRNA_chr)
    rm(ignore_chr)
}

longmask_id2 = get(load(paste0(save_path, "/LongMask/variant_ncRNA_id2_long_mask.RData")))
longmask_id7 = get(load(paste0(save_path, "/LongMask/variant_ncRNA_id7_long_mask.RData")))
longmask = cbind(longmask_id2, longmask_id7)
ncRNA_all = cbind(ncRNA_all, longmask)
longmask_ignore = get(load(paste0(save_path, "/LongMask/ignore_rare_gene_variant_ncRNA_long_mask.RData"))) 
ignore_all = rbind(ignore_all, longmask_ignore)

save(ncRNA_all, file=paste0(save_path, "/Merge/variant_ncRNA_all.RData")) 
save(ignore_all, file=paste0(save_path, "/Merge/ignore_rare_gene_variant_ncRNA_all.RData"))

data_array <- as.matrix(ncRNA_all)
np$savez(paste0(save_path, "/NPY_FILE/variant_ncRNA_all.npz"),
        data = data_array,
        colnames = colnames(rdata_coding),
        sampleid = rownames(rdata_coding))