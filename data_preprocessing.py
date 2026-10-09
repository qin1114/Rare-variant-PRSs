"""
python data_preprocessing.py -cutoff 0.001 -rare_gene Coding -pheno_name X30610 -trait Continuous -save_dir ../../data -split_dir ../../data/split/ -data_dir ../../STAARpipeline/Step_3/cutoff_0.01/Coding -data_noncoding_dir ../../data/STAARpipeline/Step_3/cutoff_0.01/Noncoding/NPY_FILE -cov_file ../../data/Biochemistry_matched_data.csv
"""
import argparse
import numpy as np
import torch
import pandas as pd

import sys
sys.path.append('../')
from utils.util import cal_correlation, get_res_train, nets_zscore


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    ## Training
    parser.add_argument('-data_dir', default=None, required=True, help='folder of coding burden score')
    parser.add_argument('-data_noncoding_dir', default=None, required=True, help='folder of noncoding burden score')
    parser.add_argument('-cutoff', default=0.01, required=True, help='MAF threshold for burden score')
    parser.add_argument('-rare_gene', default="Coding", required=True, choices=["Coding", "Noncoding"], help='input type of rare gene burden score')
    parser.add_argument('-pheno_name', default='X30610', required=True, help='Name of phenotype or ICD code of disease')
    parser.add_argument('-trait', default='Continuous', choices=['Continuous', 'Binary'], required=True)
    parser.add_argument('-save_dir', default=None, required=True, help='output folder')
    parser.add_argument('-split_dir', default=None, required=True, help='folder of data split')
    parser.add_argument('-z_thre', default=3.0, help='z threshold for feature pre-filtering')
    parser.add_argument('-cov_file', default=None, help='filepath of covariates, required when obtain results for quantitative traits')

    return parser

parser = cli_parser()
args = parser.parse_args()

print("=" * 80)
print("Arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
print("=" * 80)


data_dir = str(args.data_dir)
data_noncoding_dir = str(args.data_noncoding_dir)
cutoff = float(args.cutoff)
rare_gene = str(args.rare_gene)
pheno_name = str(args.pheno_name)
trait = str(args.trait)
save_dir = str(args.save_dir)
split_dir = str(args.split_dir)
z_thre = float(args.z_thre)
cov_file = str(args.cov_file)



### load phenotype and sample id
if trait == 'Binary':
    y_train_df = pd.read_csv(f"{split_dir}/{pheno_name}/disease_train.txt", sep="\t")
    y_val_test_df = pd.read_csv(f"{split_dir}/{pheno_name}/disease_val_test.txt", sep="\t")
    y_train = y_train_df[pheno_name].values
    y_val_test = y_val_test_df[pheno_name].values

    train_id = y_train_df["IID"].values
    val_test_id = y_val_test_df["IID"].values

else:
    # regressed covariate before modelling, 
    # these covariate-regressed features were used for all downstream analysis, e.g. calculation of PRScs and objective of predictive rvPRS models.
    cov_data = pd.read_table(cov_file)
    cov_data = cov_data[["eid", '31-0.0', '21003-0.0', "22009-0.1", "22009-0.2", "22009-0.3", "22009-0.4", "22009-0.5", "22009-0.6", "22009-0.7", "22009-0.8", "22009-0.9", "22009-0.10"]]
    cov_data['IID'] = cov_data['eid'].astype(str) + '_' + cov_data['eid'].astype(str) 
    cov_data = cov_data.dropna().reset_index(drop=True)  
    cov_data["age2"] = cov_data["21003-0.0"]**2
    cov_feature = nets_zscore(cov_data[["21003-0.0", "age2", "31-0.0", "22009-0.1", "22009-0.2", "22009-0.3", "22009-0.4", "22009-0.5", "22009-0.6", "22009-0.7", "22009-0.8", "22009-0.9", "22009-0.10"]].values)

    y_train_df = pd.read_csv(f"{split_dir}/{pheno_name}/pheno_train.txt", sep="\t")
    y_val_test_df = pd.read_csv(f"{split_dir}/{pheno_name}/pheno_val_test.txt", sep="\t")

    train_id = y_train_df["IID"].values
    val_test_id = y_val_test_df["IID"].values
    _, ia1_train, _ = np.intersect1d(cov_data['IID'].values, train_id, return_indices=True)
    cov_feature_train = cov_feature[ia1_train]
    _, ia1_val_test, _ = np.intersect1d(cov_data['IID'].values, val_test_id, return_indices=True)
    cov_feature_test = cov_feature[ia1_val_test] 

    print(f"Pheno {pheno_name} with {len(train_id)} discovery samples and {len(val_test_id)} target samples")
    train_res, beta = get_res_train(cov_feature_train, y_train_df['pheno'].values) # beta: (14, 4)
    cov_feature2_test = np.hstack((cov_feature_test, np.ones((cov_feature_test.shape[0], 1)))) 
    test_res = y_val_test_df['pheno'].values - np.dot(cov_feature2_test, beta)

    y_mean = np.nanmean(train_res, axis=0, keepdims=True)
    y_std = np.nanstd(train_res, axis=0, keepdims=True)
    y_train = (train_res - y_mean) / y_std
    y_val_test = (test_res - y_mean) / y_std
    print(f'Beta: {beta.tolist()}, normalized with train set mean: {y_mean.item()}, std: {y_std.item()}')

    y_train_df['pheno'] = y_train
    y_val_test_df['pheno'] = y_val_test
    y_train_df.to_csv(f"{split_dir}/{pheno_name}/residual_train.txt", sep="\t", index=False)
    y_val_test_df.to_csv(f"{split_dir}/{pheno_name}/residual_val_test.txt", sep="\t", index=False)
    print(f'Successfully saved covariate-regressed phenotype to {split_dir}/{pheno_name}!')



###do chr 1-22
for chr_num in range(1,23):
    if rare_gene=="Coding":
        omic1 = np.load(f'{data_dir}/chr{chr_num}/chr_{chr_num}_all_categories_variant.npz')
        omic_colname = np.array(omic1['colnames'])
        sample_ids = np.array(omic1['sampleid'])
        omic1 = omic1['data']

        train_id, ia1_train, _ = np.intersect1d(sample_ids, train_id, return_indices=True)
        omic_data_train = omic1[ia1_train, :]
        omic_data_train[np.isnan(omic_data_train)] = 0 
        print(f'Total number of Coding matched subjects for training = {len(omic_data_train)}')

        val_test_id, ia1_val_test, _ = np.intersect1d(sample_ids, val_test_id, return_indices=True)
        omic_data_val_test = omic1[ia1_val_test, :]
        omic_data_val_test[np.isnan(omic_data_val_test)] = 0 
        print(f'Total number of Coding matched subjects for val and test = {len(omic_data_val_test)}')

    elif rare_gene=="Noncoding":
        if chr_num==1:
            omic2_ncRNA = np.load(f'{data_noncoding_dir}/variant_ncRNA_all.npz', allow_pickle=True)  
            omic2_chr1 = np.load(f'{data_noncoding_dir}/chr1_variant_Noncoding_merged.npz', allow_pickle=True) 
            omic2_longmask = np.load(f'{data_noncoding_dir}/Noncoding_allrare_variant_longmask.npz', allow_pickle=True)

            omic2 = np.hstack((omic2_ncRNA['data'], omic2_chr1['data'])) 
            omic2 = np.hstack((omic2, omic2_longmask['data'])) 
            omic_colname = np.hstack((omic2_ncRNA['colnames'], omic2_chr1['colnames']))

            omic_colname = np.hstack((omic_colname, omic2_longmask['colnames'].flatten()))
            sample_ids = np.array(omic2_chr1['sampleid'])

        else:
            omic2 = np.load(f'{data_noncoding_dir}/chr{chr_num}_variant_Noncoding_merged.npz', allow_pickle=True) 
            omic_colname = np.array(omic2['colnames'])
            sample_ids = np.array(omic2['sampleid'])
            omic2 = omic2['data']
        
        train_id, ia1_train, _ = np.intersect1d(sample_ids, train_id, return_indices=True)
        omic_data_train = omic2[ia1_train, :]
        omic_data_train[np.isnan(omic_data_train)] = 0 
        print(f'Total number of Noncoding matched subjects for training = {len(omic_data_train)}')
        
        val_test_id, ia1_val_test, _ = np.intersect1d(sample_ids, val_test_id, return_indices=True)
        omic_data_val_test = omic2[ia1_val_test, :]
        omic_data_val_test[np.isnan(omic_data_val_test)] = 0 
        print(f'Total number of Noncoding matched subjects for val and test = {len(omic_data_val_test)}')

    else:
        print(f"Illegal input type of rare gene!")


    x_train = omic_data_train
    x_val_test = omic_data_val_test
    y_train = y_train.reshape(-1, 1)
    y_val_test = y_val_test.reshape(-1, 1)


    ## compute corr and select top x% features
    rr = np.max(np.abs(cal_correlation(x_train, y_train)), axis=1).flatten()
    rr[np.isnan(rr)] = 0.0
    z_value = rr * np.sqrt((x_train.shape[0]-2)/(1-rr**2))
    
    shift_indx = np.argsort(z_value)
    z_value_sorted = np.sort(z_value)
    shift_indx = shift_indx[np.abs(z_value_sorted)>z_thre]
    print(z_value[shift_indx])
    print(omic_colname[shift_indx])

    print(f'Number of feature selected for pheno {pheno_name} of chr {chr_num}: = {len(shift_indx)} / {len(z_value)}')


    ## Reorder according to correlation
    if chr_num==1:
        x_train_for_dataloader = x_train[:,shift_indx]
        x_val_test_for_dataloader = x_val_test[:,shift_indx]
        z_value_features = z_value[shift_indx]
        columns_name = omic_colname[shift_indx]

        omic_colname_total = omic_colname 
        z_value_total = z_value

    else:
        x_train_for_dataloader = np.hstack((x_train_for_dataloader, x_train[:,shift_indx]))
        x_val_test_for_dataloader = np.hstack((x_val_test_for_dataloader, x_val_test[:,shift_indx]))
        z_value_features = np.concatenate((z_value_features, z_value[shift_indx]))
        columns_name = np.hstack((columns_name, omic_colname[shift_indx]))

        omic_colname_total = np.hstack((omic_colname_total, omic_colname))
        z_value_total = np.hstack((z_value_total, z_value))


save_file = f'{save_dir}/{rare_gene}/rare_data_for_{pheno_name}_{cutoff}.npz'
np.savez(save_file, 
    x_train_for_dataloader=x_train_for_dataloader, x_val_test_for_dataloader=x_val_test_for_dataloader, 
    y_train=y_train, y_val_test=y_val_test, train_id=train_id, val_test_id=val_test_id,
    z_value_features=z_value_features, columns_name=columns_name, omic_colname_total=omic_colname_total, z_value_total=z_value_total)
print(f'Number of feature selected for pheno {pheno_name}: = {x_train_for_dataloader.shape[1]}')


