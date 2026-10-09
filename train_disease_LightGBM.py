"""
python train_disease_LightGBM.py -data_dir ../../data/ -cutoff 0.01 -disease_select E4_DM2 -rare_gene Coding -split_dir ../../data/split -save_dir ../../results/LightGBM
"""
import random
import numpy as np
import pandas as pd
import torch
import os
import argparse
import copy
from lightgbm import LGBMClassifier, log_evaluation
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import json, joblib
import pickle
from itertools import product



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
print(x_train_for_dataloader.shape, y_train.shape)


y_train = y_train.ravel().astype(int)
y_val = y_val.ravel().astype(int)
y_test= y_test.ravel().astype(int)


param_grid = {
    'learning_rate': [0.5, 0.2, 0.1, 0.05],
    'num_leaves': [15, 31, 63, 127, 255],
    'min_child_samples': [50, 100, 200],
    "max_depth": [-1],
}

def select_params_combo(my_dict, nb_items, my_seed):
    combo_list = [dict(zip(my_dict.keys(), v)) for v in product(*my_dict.values())]
    random.seed(my_seed)
    return random.sample(combo_list, nb_items)

nb_params = 30 
my_seed = 2024

candidate_params_lst = select_params_combo(param_grid, nb_params, my_seed)


best_auc = -1
model_params_all = {}
index = 1
for i in range(0, len(candidate_params_lst)):

    current_params = {}
    current_params.update(candidate_params_lst[i].copy())
    print(current_params)
    
    model = LGBMClassifier(objective='binary', 
            n_estimators=1000,
            class_weight='balanced', 
            boosting_type='gbdt',
            seed=2024,
            verbose=-1, 
            early_stopping_rounds=100) 
    model.set_params(**current_params)
    
    model.fit(
        x_train_for_dataloader,
        y_train,
        eval_set=[(x_val_for_dataloader, y_val), (x_train_for_dataloader, y_train)],
        eval_metric=['aucpr', 'auc', 'binary_logloss'],
        callbacks=[
            log_evaluation(period=50) 
        ]
    )


    val_pred = model.predict_proba(x_val_for_dataloader)[:, 1] 
    val_pred_label = (val_pred >= 0.5).astype(int)
    
    val_auc = roc_auc_score(y_val, val_pred)
    val_acc = balanced_accuracy_score(y_val, val_pred_label)

    model_params = {}
    model_params["val_best_ce"] = model.evals_result_['valid_0']['binary_logloss']
    model_params["val_best_auc"] = model.evals_result_['valid_0']['auc']
    model_params["best_epoch"] = model.best_iteration_
    model_params_all[f"model{index}"] = model_params
    
    if val_auc >= best_auc:
        best_auc = val_auc
        best_params = current_params
    
    index = index+1
    print(f"val acc: {val_acc}, auc: {val_auc}.")

    pred_all = model.predict_proba(x_test_for_dataloader)[:, 1]  
    pred_all_label = (pred_all >= 0.5).astype(int)

    test_acc = balanced_accuracy_score(y_test, pred_all_label)
    test_auc = roc_auc_score(y_test, pred_all)

    print(f'test best epoch = {model.best_iteration_}, prediction acc = {test_acc}, auc = {test_auc}')


print(best_params)
final_model = LGBMClassifier(
            n_estimators=1000, 
            objective='binary', 
            class_weight='balanced', 
            boosting_type='gbdt', # gbdt
            seed=2024,
            verbose=-1, 
            early_stopping_rounds=100) 
final_model.set_params(**best_params)

final_model.fit(
    x_train_for_dataloader,
    y_train,
    eval_set=[(x_val_for_dataloader, y_val), (x_train_for_dataloader, y_train)],
    eval_metric=['aucpr', 'auc', 'binary_logloss'],
    callbacks=[
        log_evaluation(period=50) 
    ]
)

pred_all = final_model.predict_proba(x_test_for_dataloader)[:, 1] 
pred_all_label = (pred_all >= 0.5).astype(int)

test_acc = balanced_accuracy_score(y_test, pred_all_label)
test_auc = roc_auc_score(y_test, pred_all)

print(f'best epoch = {final_model.best_iteration_}, prediction acc = {test_acc}, auc = {test_auc}.')


model_params = {}
model_params["best_params"] = best_params
model_params["test_acc"] = test_acc
model_params["test_auc"] = test_auc
model_params["best_epoch"] = final_model.best_iteration_
model_params_all["model_best"] = model_params

with open(f'{save_dir}/models/{rare_gene}/{disease_select}_cutoff{cutoff}_model_params.json', 'w') as f:
    json.dump(model_params_all, f)
joblib.dump(final_model, f'{save_dir}/models/{rare_gene}/{disease_select}_cutoff{cutoff}_lightgbm_model.joblib')


train_pred_all = final_model.predict_proba(x_train_for_dataloader)[:, 1] 
val_pred_all = final_model.predict_proba(x_val_for_dataloader)[:, 1]

res_df = pd.DataFrame({
    "IID": np.concatenate([train_id, val_id, test_id]),
    "disease": np.concatenate([y_train.flatten(), y_val.flatten(), y_test.flatten()]),
    "Probability": np.concatenate([train_pred_all, val_pred_all, pred_all]),
    "dataset": np.array(["train"]*len(y_train) + ["val"]*len(y_val) + ["test"]*len(y_test)),
})
res_df.to_csv(f"{save_dir}/PRS/{rare_gene}/LightGBM_PRS_{disease_select}_cutoff{cutoff}.txt", sep="\t", index=False)