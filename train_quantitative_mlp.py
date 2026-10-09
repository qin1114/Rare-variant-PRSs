"""
python train_quantitative_mlp.py -cutoff 0.01 -rare_gene Coding -device cuda:0 -pheno_name X30610 -data_dir ../../data/ -split_dir ../../data/split -save_dir ../../results/MLP
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, tensor
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import os
import pandas as pd
import argparse
import copy
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('../code/')
from utils.mlp import MLP
from utils.dataloader import load_matrix_data
from utils.loss import compute_loss_multi_mse


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    ## Training
    parser.add_argument('-lr', default=1e-4, help='Learning rate')
    parser.add_argument('-batch_size', default=64, help='Batch size')
    parser.add_argument('-nepoch', default=30, help='Number of epochs')
    parser.add_argument('-seed', default=42, help='Number of epochs')
    ## model
    parser.add_argument('-accumulation_step', default=None, help='Accumulation step, default to loop across [8, 32]')
    parser.add_argument('-hidden_dim', default=None, help='Network depth, default to loop across [128, 256, 512]')
    parser.add_argument('-input_drop', default=0.25, help='Input dropout')
    parser.add_argument('-pred_drop', default=0.5, help='Prediction layer dropout')
    ## data
    parser.add_argument('-data_dir', default=None, required=True, help='folder of burden score')
    parser.add_argument('-cutoff', default=0.01, required=True, help='MAF threshold for burden score')
    parser.add_argument('-pheno_name', default='X30610', required=True, help='Name of phenotype')
    parser.add_argument('-rare_gene', default="Coding", required=True, choices=["Coding", "Noncoding"], help='input type of rare gene burden score')
    parser.add_argument('-split_dir', default=None, required=True, help='folder of data split')
    parser.add_argument('-z_thre', default=3.29, help='z threshold for feature pre-filtering')
    ## others
    parser.add_argument('-save_dir', default=None, required=True, help='output folder')
    parser.add_argument('-device', default="cuda:0", required=True, help='cuda')
    parser.add_argument('-renormalize', default=False, help='use_checkpoint')
    parser.add_argument('-zscore_features', default=False, help='use_checkpoint')
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
seed = int(args.seed)
## model
accumulation_steps = [8, 32] if args.accumulation_step is None else [args.accumulation_step]
hidden_dim = [128, 256, 512] if args.hidden_dim is None else [args.hidden_dim]
input_drop = float(args.input_drop)
pred_drop = float(args.pred_drop)
## data
data_dir = str(args.data_dir)
cutoff = float(args.cutoff) 
pheno_name = str(args.pheno_name)
rare_gene = str(args.rare_gene)
split_dir = str(args.split_dir)
z_thre = float(args.z_thre)
## others
save_dir = str(args.save_dir)
renormalize = bool(args.renormalize)
zscore_features = bool(args.zscore_features)


save_name = f'Rare_{rare_gene}_cutoff{cutoff}{pheno_name}_{z_thre}_{lr}_{nepoch}'
os.system('mkdir -p ' + f"{save_dir}/results/{rare_gene}/{save_name}")
os.system('mkdir -p ' + f'{save_dir}/models/{rare_gene}/{save_name}')
os.system('mkdir -p ' + f'{save_dir}/PRS/{rare_gene}/{pheno_name}')
os.system('mkdir -p ' + f'{save_dir}/models/{rare_gene}/runs')


params = []
for i in range(0,len(accumulation_steps)):
    for k in range(0,len(hidden_dim)):
            params.append([accumulation_steps[i], hidden_dim[k]])


def set_seed(seed=42):
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi GPU
    # NumPy
    np.random.seed(seed)
    random.seed(seed)
    # CUDA and cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(seed)



## Load data
if rare_gene == "Coding":

    saved_pheno_data = f'{data_dir}/Coding/rare_data_for_{pheno_name}_{cutoff}.npz'
    info_all = np.load(saved_pheno_data, allow_pickle=True)
    x_train_for_dataloader = info_all['x_train_for_dataloader']
    x_val_test_for_dataloader = info_all['x_val_test_for_dataloader']
    val_test_id = info_all["val_test_id"]

    y_train = torch.FloatTensor(info_all['y_train'].reshape(-1, 1))
    y_val_test = torch.FloatTensor(info_all['y_val_test'].reshape(-1, 1))
    z_value_features = info_all['z_value_features']
    columns_name = info_all["columns_name"]

elif rare_gene == "Noncoding":

    saved_pheno_data = f'{data_dir}/Noncoding/rare_data_for_{pheno_name}_{cutoff}.npz'
    info_all = np.load(saved_pheno_data, allow_pickle=True)
    x_train_for_dataloader = info_all['x_train_for_dataloader']
    x_val_test_for_dataloader = info_all['x_val_test_for_dataloader']
    val_test_id = info_all["val_test_id"]
    
    y_train = torch.FloatTensor(info_all['y_train'].reshape(-1, 1))
    y_val_test = torch.FloatTensor(info_all['y_val_test'].reshape(-1, 1))
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

freq_train = np.mean(x_train_for_dataloader) 
print(f"genes frequancy: {freq_train} for train dataloader!")


x_train_for_dataloader = x_train_for_dataloader[:,indx_resort]
x_val_for_dataloader = x_val_for_dataloader[:,indx_resort]
x_test_for_dataloader = x_test_for_dataloader[:,indx_resort]


if zscore_features:
    x_mean = np.mean(x_train_for_dataloader, axis=0)
    x_std= np.std(x_train_for_dataloader, axis=0)
    x_train_for_dataloader = (x_train_for_dataloader - x_mean)/x_std
    x_val_for_dataloader = (x_train_for_dataloader - x_mean)/x_std
    x_test_for_dataloader = (x_test_for_dataloader - x_mean)/x_std


train_dataset = load_matrix_data(x = x_train_for_dataloader, y=y_train, cov_fea=None)
ukb_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
print(f"train loader length: {len(ukb_loader)}")
niter = len(ukb_loader)
val_dataset = load_matrix_data(x = x_val_for_dataloader, y=y_val, cov_fea=None)
ukb_loader_val = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
print(f"val loader length: {len(ukb_loader_val)}")
test_dataset = load_matrix_data(x = x_test_for_dataloader, y=y_test, cov_fea=None)
ukb_loader_test = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)
print(f"test loader length: {len(ukb_loader_test)}")

print(f'Number of features = {x_train_for_dataloader.shape[1]}')
pheno_max = max(torch.max(y_train), torch.max(y_val_test))
pheno_min = min(torch.min(y_train), torch.min(y_val_test))
print(f'Range of phenotype = [{pheno_min}, {pheno_max}]')



for jjj in range(0, len(params)):

    save_ex = f"{params[jjj][0]}_{params[jjj][1]}"
    accumulation_step = params[jjj][0]
    print(params[jjj])

    ## Init tensorboard
    os.system('mkdir -p ' + f'{save_dir}/models/{rare_gene}/runs/{save_ex}')
    writer = SummaryWriter(log_dir=f'{save_dir}/models/{rare_gene}/runs/{save_ex}')
    

    mlp = MLP(in_features=x_train_for_dataloader.shape[1], hidden_features=params[jjj][1], drop=pred_drop).to(device)

    nparam = 0
    for p in mlp.parameters():
        if p.requires_grad is True:
            nparam = nparam + np.prod(p.shape)
    print('Number of param = ' + str(nparam / 1000000) + 'M')

    mlp.train()
    mlp.to(device)
    
    loss_fun = compute_loss_multi_mse(ntask = y_train.shape[1]).to(device)
       
    params11 = ([p for p in mlp.parameters()])
    optimizer = torch.optim.AdamW(params11, lr=lr, weight_decay=0.01, betas=(0.9, 0.999))

    lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,first_cycle_steps=int(nepoch * niter/accumulation_step),
                                                cycle_mult=1.0,
                                                max_lr=lr,
                                                min_lr=1e-10,
                                                warmup_steps=int(niter * 5/accumulation_step),
                                                gamma=1.0)


    val_r_ls = np.zeros((nepoch))
    test_r_ls = np.zeros((nepoch))
    best_perform = -1
    for ep in range(0, nepoch):

        optimizer.zero_grad()
        pred_all_train = np.zeros((len(y_train), y_train.shape[1]))
        for batch_idx, _batch in enumerate(ukb_loader):

            data_in = _batch[0].to(device)
            labels = torch.FloatTensor(_batch[1]).to(device)
            ## renormalize data
            if renormalize:
                data_mean = torch.mean(data_in, dim=2, keepdim=True)
                data_std = torch.std(data_in, dim=2, keepdim=True)
                data_in = (data_in - data_mean)/(data_std+1e-5)
           
            context = data_in
            pred = mlp(context)
            pred_all_train[_batch[-1].cpu().numpy(),:] = pred.detach().cpu().numpy().reshape(-1,1)
            
            loss = loss_fun(pred, labels)/accumulation_step

            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)

            if (batch_idx+1) % accumulation_step == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                optimizer.zero_grad()
                lr_scheduler.step()
                # ema(mlp_ema, mlp, 0.999)

            if batch_idx % 2000 == 0:
                loss_print = loss.detach().item()
                print('Iter: ' + str(batch_idx) + ', Loss:' + str(loss_print))


        writer.add_scalar('Loss/train', loss.detach().item(), ep)
        lr_cur = optimizer.param_groups[0]['lr']
        writer.add_scalar('lr/train', lr_cur, ep)
        
        mlp.eval() 
        if ep == (nepoch-1):
            torch.save(mlp.state_dict(), f'{save_dir}/models/{rare_gene}/' + 'model_ep' + str(ep) + f"_{save_ex}" + '.pth')
        

        # val: find best epoch model
        with torch.no_grad():
            pred_all = np.zeros((len(y_val), y_train.shape[1]))
            for batch_idx, _batch in enumerate(ukb_loader_val):
                data_in = _batch[0].to(device)
                ## renormalize data
                if renormalize:
                    data_mean = torch.mean(data_in, dim=2, keepdim=True)
                    data_std = torch.std(data_in, dim=2, keepdim=True)
                    data_in = (data_in - data_mean)/(data_std+1e-5)

                pred = mlp(data_in)
                pred_all[_batch[-1].cpu().numpy(),:] = pred.detach().cpu().numpy().reshape(-1,1)

                if batch_idx==0:
                    id_index = _batch[2].numpy()
                else:
                    id_index = np.concatenate((id_index, _batch[2].numpy()))

            val_r = np.corrcoef(pred_all.flatten(), y_val.numpy().flatten())[0,1] 
            val_r_ls[ep] = val_r
            writer.add_scalar('ACC/R', val_r, ep)
            print(f'Epoch: {ep}, val r = {val_r}')

            ####save the best model
            if best_perform <float(val_r):
                best_perform = float(val_r) * 1
                best_model = copy.deepcopy(mlp)
                best_ep = ep
                val_pred_best = pred_all
                train_pred_best = pred_all_train


        #do test for current epoch
        with torch.no_grad():

            pred_all = np.zeros((len(y_test), y_train.shape[1]))
            for batch_idx, _batch in enumerate(ukb_loader_test):
                # print(batch_idx)
                data_in = _batch[0].to(device)
                ## renormalize data
                if renormalize:
                    data_mean = torch.mean(data_in, dim=2, keepdim=True)
                    data_std = torch.std(data_in, dim=2, keepdim=True)
                    data_in = (data_in - data_mean)/(data_std+1e-5)

                pred = mlp(data_in)
                pred_all[_batch[-1].cpu().numpy(),:] = pred.detach().cpu().numpy().reshape(-1,1)

                if batch_idx==0:
                    id_index = _batch[2].numpy()
                else:
                    id_index = np.concatenate((id_index, _batch[2].numpy()))

            test_r = np.corrcoef(pred_all.flatten(), y_test.numpy().flatten())[0,1] 
            writer.add_scalar('R/test', test_r, ep)
            print(f'Epoch: {ep}, test r = {test_r}')
            test_r_ls[ep] = test_r

        mlp.train() 

    np.save(f'{save_dir}/results/{rare_gene}/{save_name}/mlp_{save_ex}_val_r.npy', val_r_ls)
    np.save(f'{save_dir}/results/{rare_gene}/{save_name}/mlp_{save_ex}_test_r.npy', test_r_ls)
    torch.save(best_model.state_dict(), f'{save_dir}/models/{rare_gene}' + 'model_best_ep' + str(best_ep) + f"_{save_ex}" + '.pth')

       

    #do final test: model with best epoch
    with torch.no_grad():

        best_model.eval()
        pred_all = np.zeros((len(y_test), y_train.shape[1]))
        for batch_idx, _batch in enumerate(ukb_loader_test):
            # print(batch_idx)
            data_in = _batch[0].to(device)
            ## renormalize data
            if renormalize:
                data_mean = torch.mean(data_in, dim=2, keepdim=True)
                data_std = torch.std(data_in, dim=2, keepdim=True)
                data_in = (data_in - data_mean)/(data_std+1e-5)

            pred = best_model(data_in)
            pred_all[_batch[-1].cpu().numpy(),:] = pred.detach().cpu().numpy().reshape(-1,1)

            if batch_idx==0:
                id_index = _batch[2].numpy()
            else:
                id_index = np.concatenate((id_index, _batch[2].numpy()))
            
        test_r = np.corrcoef(pred_all.flatten(), y_test.numpy().flatten())[0,1]
        print(f'{pheno_name}: epoch {best_ep}, test r = {test_r}')


    writer.close()


    ## Save PRS
    res_df = pd.DataFrame({
        "IID": np.concatenate([train_id, val_id, test_id]),
        "pheno": np.concatenate([y_train.flatten(), y_val.flatten(), y_test.flatten()]),
        "SCORE": np.concatenate([train_pred_best.flatten(), val_pred_best.flatten(), pred_all.flatten()]),
        "dataset": np.array(["train"]*len(y_train) + ["val"]*len(y_val) + ["test"]*len(y_test)),
    })
    res_df.to_csv(f"{save_dir}/PRS/{rare_gene}/{pheno_name}/MLP_PRS_{save_ex}_cutoff{cutoff}.txt", sep="\t", index=False)


