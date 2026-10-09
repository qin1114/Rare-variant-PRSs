
"""
python visual_RVTrans_saliency_disease.py -accumulation_step 8 -patch_size 64 -depth 6 -cutoff 0.01 -rare_gene Coding -device cuda:0 -disease_select E4_DM2 -data_dir ../../data/ -split_dir ../../data/split -save_dir ../../results/RVTrans
"""
import pandas as pd
import numpy as np
import torch
import os
import argparse
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('../')
from utils.model_disease import RVTrans
from utils.dataloader import load_matrix_data



def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    ## Training
    parser.add_argument('-lr', default=1e-4, help='Learning rate')
    parser.add_argument('-batch_size', default=64, help='Training batch size')
    parser.add_argument('-nepoch', default=30, help='Number of epochs')
    ## model
    parser.add_argument('-accumulation_step', required=True, default=None, help='Accumulation step')
    parser.add_argument('-depth', required=True, default=None, help='Network depth')
    parser.add_argument('-patch_size', required=True, default=None, help='patch_size')
    parser.add_argument('-output_dim', default=2)
    parser.add_argument('-embed_dim', default=512, help='embed_dim')
    parser.add_argument('-emb_dim1', default=256, help='emb_dim1')
    parser.add_argument('-num_heads', default=8, help='Number of head')
    parser.add_argument('-input_drop', default=0.25, help='Input dropout')
    parser.add_argument('-pred_drop', default=0.5, help='Prediction layer dropout')
    ## data
    parser.add_argument('-data_dir', default=None, required=True, help='folder of burden score')
    parser.add_argument('-cutoff', default=0.01, required=True, help='MAF threshold for burden score')
    parser.add_argument('-disease_select', default='E4_DM2', required=True, help='ICD10 code of disease')
    parser.add_argument('-rare_gene', default="Coding", required=True, choices=["Coding", "Noncoding"], help='input type of rare gene burden score')
    parser.add_argument('-split_dir', default=None, required=True, help='folder of data split')
    parser.add_argument('-z_thre', default=3.29, help='z threshold for feature pre-filtering')
    ## others
    parser.add_argument('-save_dir', default=None, required=True, help='output folder')
    parser.add_argument('-device', default="cuda:0", required=True, help='cuda')
    return parser


parser = cli_parser()
args = parser.parse_args()

print("=" * 80)
print("Arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
print("=" * 80)

device = str(args.device) if torch.cuda.is_available() else 'cpu'
print("Let's use", device, "GPU!")

## Training
lr = float(args.lr)
batch_size = int(args.batch_size)
nepoch = int(args.nepoch)
## model
accumulation_step = int(args.accumulation_step)
depth = int(args.depth)
patch_size = int(args.patch_size)

output_dim = int(args.output_dim)
embed_dim = int(args.embed_dim)
emb_dim1 = int(args.emb_dim1)
num_heads = int(args.num_heads)
input_drop = float(args.input_drop)
pred_drop = float(args.pred_drop)
## data
data_dir = str(args.data_dir)
cutoff = float(args.cutoff) 
disease_select = str(args.disease_select)
rare_gene = str(args.rare_gene)
split_dir = str(args.split_dir)
z_thre = float(args.z_thre)
## others
save_dir = str(args.save_dir)

os.system('mkdir -p ' + f"{save_dir}/saliency/{rare_gene}")


## Load data
if rare_gene == "Coding":

    saved_pheno_data = f'{data_dir}/Coding/rare_data_for_{disease_select}_{cutoff}.npz'
    info_all = np.load(saved_pheno_data, allow_pickle=True)
    x_train_for_dataloader = info_all['x_train_for_dataloader']
    x_val_test_for_dataloader = info_all['x_val_test_for_dataloader']
    val_test_id = info_all["val_test_id"]

    y_train = torch.FloatTensor(info_all['y_train'].reshape(-1, 1))
    y_val_test = torch.FloatTensor(info_all['y_val_test'].reshape(-1, 1))
    z_value_features = info_all['z_value_features']
    columns_name = info_all["columns_name"]

elif rare_gene == "Noncoding":

    saved_pheno_data = f'{data_dir}/Noncoding/rare_data_for_{disease_select}_{cutoff}.npz'
    info_all = np.load(saved_pheno_data, allow_pickle=True)
    x_train_for_dataloader = info_all['x_train_for_dataloader']
    x_val_test_for_dataloader = info_all['x_val_test_for_dataloader']
    val_test_id = info_all["val_test_id"]
    
    y_train = torch.FloatTensor(info_all['y_train'].reshape(-1, 1))
    y_val_test = torch.FloatTensor(info_all['y_val_test'].reshape(-1, 1))
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


freq_train = np.mean(x_train_for_dataloader) 
print(f"genes frequancy: {freq_train} for train dataloader!")

x_train_for_dataloader = x_train_for_dataloader[:,indx_resort]
x_val_for_dataloader = x_val_for_dataloader[:,indx_resort]
x_test_for_dataloader = x_test_for_dataloader[:,indx_resort]



test_dataset = load_matrix_data(x = x_test_for_dataloader, y=y_test, cov_fea=None)
ukb_loader_test = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
print(f"test loader length: {len(ukb_loader_test)}")

print(f'Number of features = {x_train_for_dataloader.shape[1]}')
ratio = torch.sum(y_train) / len(y_train)
print(f'Disease ratio = {ratio}')
    

save_ex = f"{accumulation_step}_{patch_size}_{depth}"
ckpt_path = f'{save_dir}/models/{rare_gene}/Rare_{rare_gene}_cutoff{cutoff}_{disease_select}_{z_thre}_{lr}_{nepoch}_{embed_dim}_{num_heads}/'

if not os.path.exists(ckpt_path + f'model_ep{nepoch-1}' + f"_{save_ex}" + '.pth'):
    print(f"Found no model for {disease_select}")
    sys.exit(1)
else:
    import re
    file_list = os.listdir(ckpt_path)
    re_ex = save_ex.replace("2.0", "2\.0")

    pattern = fr"model_best_ep\d+_{re_ex}\.pth"
    matched_files = [f for f in file_list if re.fullmatch(pattern, f)]
    print(matched_files)
    checkpoint_path = ckpt_path + matched_files[0]
    print(checkpoint_path)
    ep = int(matched_files[0].split("_")[2][2:])        
    if len(matched_files)!=1:
        print(f"{len(matched_files)} matched files!")


model = RVTrans(input_size=x_train_for_dataloader.shape[1], patch_size=patch_size, in_chans = 1, out_chans=1, embed_dim=embed_dim, 
        depth=depth, num_heads=num_heads,mlp_ratio=4.,qkv_bias=False, qk_scale=None, 
        norm_layer=torch.nn.LayerNorm, mlp_time_embed=False, 
        use_checkpoint=False, conv=True, skip=True, 
        loss_type='ce', 
        attn_drop=0.1,
        proj_drop=0.1, 
        pred_drop=0.5,
        out_class=output_dim)


state_dic = torch.load(checkpoint_path, 'cpu', weights_only=True)
xx, yy = model.load_state_dict(state_dic, strict=True)

for param in model.parameters():
    param.requires_grad = False
model.to(device)   
model.eval()

nsub = len(x_test_for_dataloader)
slc_all = np.zeros((nsub, x_test_for_dataloader.shape[1]))
risk_all = np.zeros((nsub, 1))

for batch_idx, _batch in enumerate(ukb_loader_test):
    print(disease_select, batch_idx)
        
    data_in = _batch[0].to(device)
    data_in.requires_grad = True

    pred = model(data_in)
    prob = torch.nn.functional.softmax(pred, dim=1)

    # extract gradient for first person
    risk_score = prob[0, 1]
    risk_score.backward(retain_graph=True)
    
    slc = data_in.grad[0][0] # (1,1,feature_dim)
    slc = (slc - slc.mean())/slc.std()
    # print(slc)
    
    slc_all[batch_idx] = slc.detach().cpu().numpy()
    risk_all[batch_idx] = risk_score.detach().cpu().numpy()

    data_in.grad.zero_()


np.savez(f"{save_dir}/saliency/{rare_gene}/{disease_select}_{cutoff}.npz", 
        slc_all=slc_all, 
        risk_all=risk_all,
        feature_names=columns_name[indx_resort],
        sampleid=test_id)


weight_df = pd.DataFrame({"columns": columns_name[indx_resort], "weight": np.mean(slc_all, axis=0)})
if rare_gene == "Coding":
    weight_df[['gene', 'annotation']] = weight_df['columns'].str.split(':', expand=True)
else:
    weight_df[['gene', 'annotation']] = weight_df['columns'].str.split('_', n=1, expand=True)
df_sorted = weight_df.sort_values(by='weight', key=abs, ascending=False)

df_sorted.to_csv(f"{save_dir}/saliency/{rare_gene}/{disease_select}_{cutoff}.csv", sep="\t", index=False)
feature_names = columns_name[indx_resort]


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
    feature_names = np.array([s.replace('_', ':', 1) for s in feature_names])
    feature_names = np.array([
        f'{gene} ({mapping.get(mask, mask)})'
        for gene, mask in [f.split(':') for f in feature_names]
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
    feature_names = np.array([
        f'{gene} ({mapping.get(mask, mask)})'
        for gene, mask in [f.split(':') for f in feature_names]
    ])
    

weight_mean = (np.mean(slc_all, axis=0))
top10_indices = np.argsort(-np.abs(weight_mean))[:10] 
top10_features = feature_names[top10_indices]
top10_weight = slc_all[:, top10_indices]
top10_weight_mean = weight_mean[top10_indices]

print("Top 10 feature for RVTrans:")
for i, feature in enumerate(top10_features):
    print(f"{i+1}. {feature}: {top10_weight_mean[i]:.4f}")


