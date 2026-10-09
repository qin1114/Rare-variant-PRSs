"""
python train_disease_GLMNET.py -data_dir ../../data/ -cutoff 0.01 -disease_select E4_DM2 -rare_gene Coding -split_dir ../../data/split -save_dir ../../results/GLMNET
"""
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import random
import numpy as np
import os
import math
import pandas as pd
import argparse
import copy
from sklearn.linear_model import LogisticRegression
import pickle
from joblib import load, dump


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-data_dir', default=None, required=True, help='folder of burden score')
    parser.add_argument('-cutoff', default=0.01, required=True, help='MAF threshold for burden score')
    parser.add_argument('-disease_select', default='E4_DM2', required=True, help='ICD code of disease')
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
disease_select = str(args.disease_select)
rare_gene = str(args.rare_gene)
split_dir = str(args.split_dir)
z_thre = float(args.z_thre)
save_dir = str(args.save_dir)

os.system('mkdir -p ' + f'{save_dir}/models/{rare_gene}')
os.system('mkdir -p ' + f'{save_dir}/PRS/{rare_gene}')



## Load data
if rare_gene == "Coding":

    saved_disease_data = f'{data_dir}/Coding/rare_data_for_{disease_select}_{cutoff}.npz'
    info_all = np.load(saved_disease_data, allow_pickle=True)
    x_train_for_dataloader = info_all['x_train_for_dataloader']
    x_val_test_for_dataloader = info_all['x_val_test_for_dataloader']
    val_test_id = info_all["val_test_id"]

    y_train = info_all['y_train'].reshape(-1, 1)
    y_val_test = info_all['y_val_test'].reshape(-1, 1)
    z_value_features = info_all['z_value_features']
    columns_name = info_all["columns_name"]

elif rare_gene == "Noncoding":

    saved_disease_data = f'{data_dir}/Noncoding/rare_data_for_{disease_select}_{cutoff}.npz'
    info_all = np.load(saved_disease_data, allow_pickle=True)
    x_train_for_dataloader = info_all['x_train_for_dataloader']
    x_val_test_for_dataloader = info_all['x_val_test_for_dataloader']
    val_test_id = info_all["val_test_id"]
    
    y_train = info_all['y_train'].reshape(-1, 1)
    y_val_test = info_all['y_val_test'].reshape(-1, 1)
    z_value_features = info_all['z_value_features']
    columns_name = info_all["columns_name"]



data_split = pd.read_csv(f"{split_dir}/{disease_select}/data_split.csv")
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



alphas = [0.0, 0.1, 0.5, 0.9, 1.0]
y_train = y_train.astype(int).flatten()
y_val = y_val.astype(int).flatten()
y_test = y_test.astype(int).flatten()


models_dict = {}
best_auc = -1
for j in range(len(alphas)):
    model = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=alphas[j],                  # 0.0=L2, 1.0=L1, else: elastic net
        C=1,            
        max_iter=500,
        random_state=42,
        n_jobs=-1,      
        class_weight='balanced'
    )

    print(x_train_for_dataloader.shape, y_train.shape)
    model.fit(x_train_for_dataloader, y_train)

    proba = model.predict_proba(x_val_for_dataloader)[:, 1]
    pred_all  = (proba >= 0.5).astype(int) 
    print(np.sum(pred_all), "of", np.sum(y_val), "val cases!")

    # acc
    val_acc = balanced_accuracy_score(y_val, pred_all)
    val_auc = roc_auc_score(y_val, proba)

    print(f'prediction for alpha={alphas[j]}: acc = {val_acc}, auc = {val_auc}')

    if val_auc>=best_auc and (not math.isnan(val_auc)):
        alphas_best = alphas[j] * 1.0
        best_auc = val_auc
    
    models_dict[f"model_{alphas[j]}"] = model
    models_dict[f"val_acc_{alphas[j]}"] = val_acc
    dump(models_dict, f'{save_dir}/models/{rare_gene}/{disease_select}_cutoff{cutoff}_model_params.joblib')



model = models_dict[f"model_{alphas_best}"]
proba = model.predict_proba(x_test_for_dataloader)[:, 1]
pred_all  = (proba >= 0.5).astype(int) 

best_acc = balanced_accuracy_score(y_test, pred_all)
best_auc = roc_auc_score(y_test, proba)

print(np.sum(pred_all), "of", np.sum(y_test), "test cases!")
print(f'{disease_select}: best prediction acc = {best_acc}, auc = {best_auc}')

models_dict["test_alpha"] = alphas_best
models_dict["test_acc"] = best_acc
models_dict["test_auc"] = best_auc
dump(models_dict, f'{save_dir}/models/{rare_gene}/{disease_select}_cutoff{cutoff}_model_params.joblib')



train_proba = model.predict_proba(x_train_for_dataloader)[:, 1]
train_pred_all  = (train_proba >= 0.5).astype(int) 
val_proba = model.predict_proba(x_val_for_dataloader)[:, 1]
val_pred_all  = (val_proba >= 0.5).astype(int) 

res_df = pd.DataFrame({
    "IID": np.concatenate([train_id, val_id, test_id]),
    "disease": np.concatenate([y_train, y_val, y_test]),
    "Probability": np.concatenate([train_proba, val_proba, proba]),
    "dataset": np.array(["train"]*len(y_train) + ["val"]*len(y_val) + ["test"]*len(y_test)),
})
res_df.to_csv(f"{save_dir}/PRS/{rare_gene}/GLMNET_PRS_{disease_select}_cutoff{cutoff}.txt", sep="\t", index=False)
