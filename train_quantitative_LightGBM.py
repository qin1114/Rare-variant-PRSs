"""
python train_quantitative_LightGBM.py -data_dir ../../data/ -cutoff 0.01 -pheno_name X30610 -rare_gene Coding -split_dir ../../data/split -save_dir ../../results/LightGBM
"""
import random
import numpy as np
import pandas as pd
import torch
import os
import argparse
import copy
import lightgbm as lgb
from lightgbm import log_evaluation, record_evaluation, early_stopping
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import json, joblib
import pickle
from itertools import product



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
print(x_train_for_dataloader.shape, y_train.shape)



train_data = lgb.Dataset(x_train_for_dataloader, label=y_train)
val_data = lgb.Dataset(x_val_for_dataloader, label=y_val, reference=train_data)

params = {
    'objective': 'regression',
    'metric': 'l2',  # MSE
    'boosting_type': 'gbdt',
    'verbose': -1,
}

param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [15, 31, 63],
    'min_data_in_leaf': [10, 20, 50],
}

eval_history = {}  
callbacks = [
    log_evaluation(period=50), 
    record_evaluation(eval_history), 
    early_stopping(stopping_rounds=20, verbose=True) 
]


best_r = -1
model_params_all = {}
index = 1
for lr in param_grid['learning_rate']:
    for leaves in param_grid['num_leaves']:
        for min_data in param_grid['min_data_in_leaf']:
            current_params = params.copy()
            current_params.update({
                'learning_rate': lr,
                'num_leaves': leaves,
                'min_data_in_leaf': min_data,
            })
            print(current_params)

            model = lgb.train(
                current_params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data], 
                valid_names=['valid'], 
                callbacks=callbacks
            )


            val_pred = model.predict(x_val_for_dataloader, num_iteration=model.best_iteration) 
            model_params = {}
            model_params["coef"] = model.model_to_string() 
            model_params["val_best_mse"] = model.best_score['valid']['l2']
            model_params["best_epoch"] = model.best_iteration
            model_params_all[f"model{index}"] = model_params

            val_r = np.corrcoef(val_pred, y_val.flatten())[0,1]
            if val_r >= best_r:
                best_r = val_r
                best_params = current_params
            
            index = index+1


print(best_params)
final_model = lgb.train(
    best_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],  
    valid_names=['valid'], 
    callbacks=callbacks
)
pred_all = final_model.predict(x_test_for_dataloader, num_iteration=final_model.best_iteration) 

rr = np.corrcoef(pred_all, y_test.flatten())[0,1]
print(f'prediction r = {rr}') 


model_params = {}
model_params["best_params"] = best_params
model_params["test_rr"] = rr
model_params["coef"] = model.model_to_string()  
model_params["val_best_mse"] = final_model.best_score['valid']['l2']
model_params["best_epoch"] = final_model.best_iteration
model_params_all["model_best"] = model_params

with open(f'{save_dir}/models/{rare_gene}/{pheno_name}_cutoff{cutoff}_model_params.json', 'w') as f:
    json.dump(model_params_all, f)
joblib.dump(final_model, f'{save_dir}/models/{rare_gene}/{pheno_name}_cutoff{cutoff}_lightgbm_model.joblib')



train_pred_all = final_model.predict(x_train_for_dataloader, num_iteration=final_model.best_iteration)
val_pred_all = final_model.predict(x_val_for_dataloader, num_iteration=final_model.best_iteration)

res_df = pd.DataFrame({
    "IID": np.concatenate([train_id, val_id, test_id]),
    "pheno": np.concatenate([y_train.flatten(), y_val.flatten(), y_test.flatten()]),
    "SCORE": np.concatenate([train_pred_all, val_pred_all, pred_all]),
    "dataset": np.array(["train"]*len(y_train) + ["val"]*len(y_val) + ["test"]*len(y_test)),
})
res_df.to_csv(f"{save_dir}/PRS/{rare_gene}/LightGBM_PRS_{pheno_name}_cutoff{cutoff}.txt", sep="\t", index=False)
 