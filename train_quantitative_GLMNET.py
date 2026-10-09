"""
python train_quantitative_GLMNET.py -data_dir ../../data/ -cutoff 0.01 -pheno_name X30610 -rare_gene Coding -split_dir ../../data/split -save_dir ../../results/GLMNET
"""
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import random
import numpy as np
import os
import math
import pandas as pd
import argparse
import copy
from glmnet import ElasticNet
import pickle
from joblib import load, dump


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-data_dir', default=None, required=True, help='folder of burden score')
    parser.add_argument('-cutoff', default=0.01, required=True, help='MAF threshold for burden score')
    parser.add_argument('-pheno_name', default='X30610', required=True, help='Name of phenotype')
    parser.add_argument('-rare_gene', default="Coding", required=True, choices=["Coding", "Noncoding"], help='input type of rare gene burden score')
    parser.add_argument('-split_dir', default=None, required=True, help='folder of data split')
    parser.add_argument('-z_thre', default=3.29, help='z threshold for feature pre-filtering')
    parser.add_argument('-save_dir', default=None, required=True, help='output folder')

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
pheno_name = str(args.pheno_name)
rare_gene = str(args.rare_gene)
split_dir = str(args.split_dir)
z_thre = float(args.z_thre)
save_dir = str(args.save_dir)

os.system('mkdir -p ' + f'{save_dir}/models/{rare_gene}')
os.system('mkdir -p ' + f'{save_dir}/PRS/{rare_gene}')



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



alphas = [0.0, 0.01, 0.1, 0.5, 0.8, 0.99, 1.0]
y_train = y_train.astype(int).flatten()
y_val = y_val.astype(int).flatten()
y_test = y_test.astype(int).flatten()

models_dict = {}
best_r = -1
for j in range(len(alphas)):
    model = ElasticNet(alpha=alphas[j], n_lambda=30, random_state=1024)
    print(x_train_for_dataloader.shape, y_train.shape)
    model.fit(x_train_for_dataloader, y_train)

    pred_all = model.predict(x_val_for_dataloader) 
    rr = np.corrcoef(pred_all,y_val)[0,1]
    print(f'prediction for alpha={alphas[j]}: rr = {rr}')

    if rr>=best_r and (not math.isnan(rr)):
        alphas_best = alphas[j] * 1.0
        best_r = rr
    
    models_dict[f"model_{alphas[j]}"] = model
    models_dict[f"val_r_{alphas[j]}"] = best_r
    dump(models_dict, f'{save_dir}/models/{rare_gene}/{pheno_name}_cutoff{cutoff}_model_params.joblib')


model = models_dict[f"model_{alphas_best}"]
pred_all = model.predict(x_test_for_dataloader) 
best_r = np.corrcoef(pred_all,y_test)[0,1]
print(f'{pheno_name}: best prediction r = {best_r}')

models_dict["test_alpha"] = alphas_best
models_dict["test_r"] = best_r
dump(models_dict, f'{save_dir}/models/{rare_gene}/{pheno_name}_cutoff{cutoff}_model_params.joblib')


train_pred_all = model.predict(x_train_for_dataloader) 
val_pred_all = model.predict(x_val_for_dataloader) 

res_df = pd.DataFrame({
    "IID": np.concatenate([train_id, val_id, test_id]),
    "pheno": np.concatenate([y_train, y_val, y_test]),
    "SCORE": np.concatenate([train_pred_all, val_pred_all, pred_all]),
    "dataset": np.array(["train"]*len(y_train) + ["val"]*len(y_val) + ["test"]*len(y_test)),
})
res_df.to_csv(f"{save_dir}/PRS/{rare_gene}/GLMNET_PRS_{pheno_name}_cutoff{cutoff}.txt", sep="\t", index=False)
