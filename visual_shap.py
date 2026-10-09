"""
python visual_shap.py -data_dir ../../data/ -cutoff 0.01 -pheno_name X30610 -rare_gene Coding -split_dir ../../data/split -save_dir ../../results/ -model_file ../../results/LightGBM/
"""
import lightgbm as lgb
import joblib
import shap
import numpy as np
import pandas as pd
import os
import sys
import argparse


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    ## Training
    parser.add_argument('-data_dir', default=None, required=True, help='folder of burden score')
    parser.add_argument('-cutoff', default=0.01, required=True, help='MAF threshold for burden score')
    parser.add_argument('-rare_gene', default="Coding", required=True, choices=["Coding", "Noncoding"], help='input type of rare gene burden score')
    parser.add_argument('-pheno_name', default='X30610', required=True, help='Name of phenotype or ICD code of disease')
    parser.add_argument('-save_dir', default=None, required=True, help='output folder')
    parser.add_argument('-model_file', default=None, required=True, help='folder of rvPRS')
    parser.add_argument('-split_dir', default=None, required=True, help='folder of data split')
    parser.add_argument('-z_thre', default=3.29, help='z threshold for feature pre-filtering')
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
save_dir = str(args.save_dir)
model_file = str(args.model_file)
split_dir = str(args.split_dir)
z_thre = float(args.z_thre)


## Load LightGBM model
model_path = f'{model_file}/models/{rare_gene}/{pheno_name}_cutoff{cutoff}_lightgbm_model.joblib'
model = joblib.load(model_path)


## Load data
if rare_gene == "Coding":

    saved_pheno_data = f'{data_dir}/Coding/rare_data_for_{pheno_name}_{cutoff}.npz'
    info_all = np.load(saved_pheno_data, allow_pickle=True)
    x_train_for_dataloader = info_all['x_train_for_dataloader']
    x_val_test_for_dataloader = info_all['x_val_test_for_dataloader']
    val_test_id = info_all["val_test_id"]

    y_train = info_all['y_train'].reshape(-1, 1)
    y_val_test = info_all['y_val_test'].reshape(-1, 1)
    z_value_features = info_all['z_value_features']
    columns_name = info_all["columns_name"]

elif rare_gene == "Noncoding":

    saved_pheno_data = f'{data_dir}/Noncoding/rare_data_for_{pheno_name}_{cutoff}.npz'
    info_all = np.load(saved_pheno_data, allow_pickle=True)
    x_train_for_dataloader = info_all['x_train_for_dataloader']
    x_val_test_for_dataloader = info_all['x_val_test_for_dataloader']
    val_test_id = info_all["val_test_id"]
    
    y_train = info_all['y_train'].reshape(-1, 1)
    y_val_test = info_all['y_val_test'].reshape(-1, 1)
    z_value_features = info_all['z_value_features']
    columns_name = info_all["columns_name"]


data_split = pd.read_csv(f"{split_dir}/{pheno_name}/data_split.csv")
train_id = data_split[data_split["split"]=="train"]["eid"].values
train_id = [f"{x}_{x}" for x in train_id]

val_id = data_split[data_split["split"]=="val"]["eid"].values
val_id = [f"{x}_{x}" for x in val_id]
test_id = data_split[data_split["split"]=="test"]["eid"].values
test_id = [f"{x}_{x}" for x in test_id]

_, ia1_val, _ = np.intersect1d(val_test_id, val_id, return_indices=True)
x_val_for_dataloader = x_val_test_for_dataloader[ia1_val, :]
y_val = y_val_test[ia1_val]

_, ia1_test, _ = np.intersect1d(val_test_id, test_id, return_indices=True)
x_test_for_dataloader = x_val_test_for_dataloader[ia1_test, :]
y_test = y_val_test[ia1_test]


## Feature filter
indx_resort = np.argsort(-z_value_features)
zthre_select = [z_value >= z_thre for z_value in z_value_features[indx_resort]]
indx_resort = indx_resort[zthre_select]

if rare_gene == "Coding":
    columns_name_select = [any(sub in column for sub in ["plof","plof_ds","missense","disruptive_missense","synonymous","ptv", "ptv_ds"]) and ("all_categories_incl_ptv" not in column) for column in columns_name[indx_resort]]
    indx_resort = indx_resort[columns_name_select]
elif rare_gene == "Noncoding":
    columns_name_select = [any(sub in column for sub in ["promoter_CAGE","promoter_DHS","enhancer_CAGE","enhancer_DHS", "downstream","upstream","UTR","ncRNA"]) for column in columns_name[indx_resort]]
    indx_resort = indx_resort[columns_name_select]
print(columns_name[indx_resort])


x_train_for_dataloader = x_train_for_dataloader[:,indx_resort]
x_val_for_dataloader = x_val_for_dataloader[:,indx_resort]
x_test_for_dataloader = x_test_for_dataloader[:,indx_resort]
# print(z_value_features[indx_resort])

x_train_for_dataloader[np.isnan(x_train_for_dataloader)] = 0
x_val_for_dataloader[np.isnan(x_val_for_dataloader)] = 0
x_test_for_dataloader[np.isnan(x_test_for_dataloader)] = 0
print(x_train_for_dataloader.shape, y_train.shape, x_test_for_dataloader.shape)



explainer = shap.TreeExplainer(model, approximate=True)
shap_values = explainer.shap_values(x_test_for_dataloader)
print(f"SHAP values shape: {shap_values.shape}")

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
    columns_name = np.array([s.replace('_', ':', 1) for s in columns_name])
    columns_name = np.array([
        f'{gene} ({mapping.get(mask, mask)})'
        for gene, mask in [f.split(':') for f in columns_name]
    ])

elif rare_gene == "Coding":
    mapping = {
        "plof": "pLoF",
        "plof_ds": "pLoF+D",
        "missense": "Missense",
        "disruptive_missense": "D Missense",
        "ptv": "PTVs",
        "ptv_ds": "PTVs+D",
        'synonymous': 'Synonymous'
    }
    columns_name = np.array([
        f'{gene} ({mapping.get(mask, mask)})'
        for gene, mask in [f.split(':') for f in columns_name]
    ])


feature_names = columns_name[indx_resort]
shap_mean = np.mean(shap_values, axis=0)
top10_indices = np.argsort(-np.abs(shap_mean))[:10]
top10_features = feature_names[top10_indices]
top10_shap = shap_values[:, top10_indices]
top10_shap_mean = shap_mean[top10_indices]

print("Top 10 SHAP:")
for i, feature in enumerate(top10_features):
    print(f"{i+1}. {feature}: {top10_shap_mean[i]:.4f}")
