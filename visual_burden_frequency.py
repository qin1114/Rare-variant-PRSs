"""
python visual_burden_frequency.py -model_name LightGBM -cutoff 0.01 -rare_gene Coding -data_dir ../../data/ -pheno_name X30610 -trait Continuous -model_file ../../results -cvprs_file ../results/PRS-CS_new/Continuous
"""
import pandas as pd
import numpy as np
import torch
from scipy.stats import rankdata
import os, argparse


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument('-data_dir', default=None, required=True, help='folder of burden score')
    parser.add_argument('-cutoff', default=0.01, required=True, help='MAF threshold for burden score')
    parser.add_argument('-rare_gene', default="Coding", required=True, choices=["Coding", "Noncoding"], help='input type of rare gene burden score')
    parser.add_argument('-pheno_name', default='X30610', required=True, help='Name of phenotype or ICD code of disease')
    parser.add_argument('-trait', default='Continuous', choices=['Continuous', 'Binary'], required=True)
    parser.add_argument('-model_file', default=None, required=True, help='folder of rvPRS')
    parser.add_argument('-model_name', default='LightGBM', required=True, help='Model name')
    parser.add_argument('-cvprs_file', default=None, required=True, help='folder of cvPRS')
    parser.add_argument('-prs_rate', default=0.01, help='rank cutoff for PRS')
    return parser

parser = cli_parser()
args = parser.parse_args()

print("=" * 80)
print("Arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
print("=" * 80)


data_dir = str(args.data_dir)
cutoff = float(args.cutoff)
rare_gene = str(args.rare_gene)
pheno_name = str(args.pheno_name)
trait = str(args.trait)
model_file = str(args.model_file)
model_name = str(args.model_name)
cvprs_file = str(args.cvprs_file)
prs_rate = float(args.prs_rate)



## Load rvPRS and cvPRS
df = pd.read_csv(f"{model_file}/{model_name}/PRS/{rare_gene}/{model_name}_PRS_{pheno_name}_cutoff{cutoff}.txt", sep="\t")
df = df[df['dataset']=="test"]
df.columns = ["IID", 'pheno', 'Rare_PRS', 'dataset']
common_df = pd.read_csv(f"{cvprs_file}/{pheno_name}/PRS-CS_PRS_test.txt", sep="\t")
common_PRS = common_df[["IID", "SCORE"]]
common_PRS.columns = ["IID", "Common_PRS"]
df = pd.merge(df, common_PRS, on="IID")


## Load burden score
saved_pheno_data = f'{data_dir}/{rare_gene}/rare_data_for_{pheno_name}_{cutoff}.npz'
info_all = np.load(saved_pheno_data, allow_pickle=True)

x_val_test_for_dataloader = info_all['x_val_test_for_dataloader']
val_test_id = info_all["val_test_id"]
y_val_test = torch.FloatTensor(info_all['y_val_test'].reshape(-1, 1))
z_value_features = info_all['z_value_features']
columns_name = info_all["columns_name"]

test_id = df["IID"].values
_, ia1_test, _ = np.intersect1d(val_test_id, test_id, return_indices=True)
x_test_for_dataloader = x_val_test_for_dataloader[ia1_test, :]
y_test = y_val_test[ia1_test]


## frequency of burden score among individuals within top 1% of rvPRS/cvPRS
df['rank'] = rankdata(-df['Rare_PRS'], method='average')
df['rank_common'] = rankdata(-df['Common_PRS'], method='average')
top_n = max(1, round(len(df) * prs_rate)) 

top_id = df.loc[df['rank'] <= top_n, 'IID'].values
_, ia_top, _ = np.intersect1d(test_id, top_id, return_indices=True)
top_burden = x_test_for_dataloader[ia_top, :]
# print(np.mean(top_burden, axis=0))

common_top_id = df.loc[df['rank_common'] <= top_n, 'IID'].values
_, ia_top_common, _ = np.intersect1d(test_id, common_top_id, return_indices=True)
common_top_burden = x_test_for_dataloader[ia_top_common, :]


if trait == "Binary":
    top_id = df.loc[(df['rank'] <= top_n) & (df['pheno']==1), 'IID'].values
    common_top_id = df.loc[(df['rank_common'] <= top_n) & (df['pheno']==1), 'IID'].values
else:
    top_1_percent = np.percentile(df['pheno'], 99)  # 前1%的阈值
    binary_label = np.where(df['pheno'] >= top_1_percent, 1,  0)
    top_id = df.loc[(df['rank'] <= top_n) & binary_label, 'IID'].values
    common_top_id = df.loc[(df['rank_common'] <= top_n) & binary_label, 'IID'].values

rare_identified = len(top_id)
common_identified = len(common_top_id)
overlap_identified = len(set(top_id) & set(common_top_id))
print(f'{pheno_name}: identified {rare_identified} by rvPRS, {common_identified} by cvPRS, with {overlap_identified} intersected\n.')



## calculated mean of frequancy 
top_mean = np.mean((top_burden!=0), axis=0)
common_top_mean = np.mean((common_top_burden!=0), axis=0)


if rare_gene == "Coding":
    columns_name_select = [(column.split(":")[1] in ["plof","missense","disruptive_missense","synonymous","ptv"]) for column in columns_name]
elif rare_gene == "Noncoding":
    columns_name_select = [any(sub in column for sub in ["promoter_CAGE","promoter_DHS","enhancer_CAGE","enhancer_DHS", "downstream","upstream","UTR","ncRNA"]) for column in columns_name]

columns_name = columns_name[columns_name_select]
columns_name = np.array([column.replace("disruptive", 'ds') for column in columns_name])
top_mean = top_mean[columns_name_select]
common_top_mean = common_top_mean[columns_name_select]

sorted_indices = np.argsort(-top_mean)
indices_to_label = sorted_indices[:20]

gene_names = columns_name[indices_to_label]
rare_mask_data = top_mean[indices_to_label]
common_mask_data = common_top_mean[indices_to_label]

if rare_gene == "Noncoding":
    mapping = {
        "downstream": "downstream",
        "upstream": "upstream",
        "UTR": "UTRs",
        "promoter_CAGE": "P CAGE",
        "promoter_DHS": "P DHS",
        "enhancer_CAGE": "E CAGE",
        'enhancer_DHS': 'E DHS'
    }
    gene_names = np.array([s.replace('_', ':', 1) for s in gene_names])
    gene_names = np.array([
        f'{gene} ({mapping.get(mask, mask)})'
        for gene, mask in [f.split(':') for f in gene_names]
    ])

else:
    mapping = {
        "plof": "pLoF",
        "plof_ds": "pLoF+D",
        "missense": "Missense",
        "ds_missense": "D Missense",
        "ptv": "PTVs",
        "ptv_ds": "PTVs+D",
        'synonymous': 'Synonymous'
    }
    gene_names = np.array([
        f'{gene} ({mapping.get(mask, mask)})'
        for gene, mask in [f.split(':') for f in gene_names]
    ])


print("Top 10 burden frequency:")
for i, feature in enumerate(gene_names):
    print(f"{i+1}. {feature}: rare frequency {rare_mask_data[i]:.4f}, common frequency {common_top_mean[i]:.4f}")