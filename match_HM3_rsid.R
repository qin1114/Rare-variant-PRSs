WGS_dir = '../../data/WGS_35W/'

HM3_rds <- readRDS("../../data/map.rds") # 1054330
HM3_rds <- HM3_rds %>% rename(POS = pos_hg38)

snp_num = 0
snp_common_num = 0
bim_num = 0
for(chr in 1:22){
    bim_path = paste0(WGS_dir, "Q0_unre_Caucasian_c", chr, ".bim")
    bim_data <- fread(bim_path, header = FALSE)
    colnames(bim_data) <- c("chr", "rsid", "GeneticDist", "POS", "Allele1", "Allele2")
    bim_num <- bim_num + dim(bim_data)[1]

    info <- merge(bim_data, HM3_rds, by = c("chr", "POS")) ## match pos
    cat(nrow(info), length(unique(info$POS)))

    info.match <- subset(info, Allele1 == a1 & Allele2 == a0) 
    # print(nrow(info.match))
    info.recode <- subset(info, Allele1 == a0 & Allele2 == a1)
    # print(nrow(info.recode))

    complement <- function(x) {
    switch (
        x,
        "A" = "T",
        "C" = "G",
        "T" = "A",
        "G" = "C",
        return(NA)
    )
    }
    info$C.A1 <- sapply(info$Allele1, complement)
    info$C.A2 <- sapply(info$Allele2, complement)
    info.complement <- subset(info, C.A1 == a1 & C.A2 == a0) # 303

    info.crecode <- subset(info, C.A1 == a0 & C.A2 == a1) # 15

    SNP.concat <- c(c(c(info.match$rsid.x, info.recode$rsid.x), info.complement$rsid.x), info.crecode$rsid.x) 
    SNP.cur_df <- info[info$rsid.x %in% SNP.concat, c('chr', "rsid.x", 'rsid.y', 'POS', 'Allele1', 'Allele2')]

    cat(chr, dim(SNP.cur_df)[1], length(unique(SNP.cur_df$POS)))
    snp_num = snp_num + dim(SNP.cur_df)[1]
    if(chr == 1){
        SNP.final <- SNP.cur_df
        info.final <- info
    } else {
        SNP.final <- rbind(SNP.final, SNP.cur_df)
        info.final <- rbind(info.final, info)
    }
}
cat(snp_num, snp_common_num, bim_num)
write.table(
  SNP.final,
  "../../data/SNP_HM3_matched.csv",
  quote = F,
  row.names = F,
  col.names = F
)
write.table(
  info.final,
  "../../data/SNP_HM3_matched_info.csv",
  quote = F,
  row.names = F,
  col.names = F
)
