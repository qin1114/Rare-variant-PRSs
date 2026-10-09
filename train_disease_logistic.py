"""
python train_disease_logistic.py -data_dir ../../data/ -cutoff 0.01 -disease_select E4_DM2 -rare_gene Coding -split_dir ../../data/split -save_dir ../../results/logistic
"""
import pandas as pd
import numpy as np
import argparse
import sys
import os
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression



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

y_train = y_train.flatten()
y_val = y_val.flatten()
y_test = y_test.flatten()



# model = LogisticRegression()
model = LogisticRegression(penalty=None,           # close regularization
        solver='lbfgs',       
        max_iter=500,
        random_state=42,
        class_weight='balanced')
# print(x_train_for_dataloader.shape, y_train.shape)
model.fit(x_train_for_dataloader, y_train)

probabilities = model.predict_proba(x_test_for_dataloader)[:, 1] 
y_pred = np.int32(probabilities>0.5)

test_acc = balanced_accuracy_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, probabilities)
print(f'{disease_select}: prediction {np.sum(y_pred).item()} / {np.sum(y_test).item()} cases,  acc = {test_acc}, auc: {test_auc}')


weights = model.coef_ 
model_df = pd.DataFrame({
    "weight": weights.squeeze(),
    "feature": columns_name[indx_resort]
})
model_df.to_csv(f"{save_dir}/models/{rare_gene}/logistic_{disease_select}_cutoff{cutoff}.txt", sep="\t", index=False)



## Save PRS
train_pred_all = model.predict_proba(x_train_for_dataloader)[:, 1] 
val_pred_all = model.predict_proba(x_val_for_dataloader)[:, 1] 

PRS_df = pd.DataFrame({
    "IID": np.concatenate([train_id, val_id, test_id]),
    "disease": np.concatenate([y_train, y_val, y_test]),
    "Probability": np.concatenate([train_pred_all, val_pred_all, probabilities]),
    "dataset": np.array(["train"]*len(y_train) + ["val"]*len(y_val) + ["test"]*len(y_test)),
})
PRS_df.to_csv(f"{save_dir}/PRS/{rare_gene}/logistic_PRS_{disease_select}_cutoff{cutoff}.txt", sep="\t", index=False)
