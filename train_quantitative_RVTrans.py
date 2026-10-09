"""
python train_quantitative_RVTrans.py -cutoff 0.01 -rare_gene Coding -device cuda:0 -pheno_name X30610 -data_dir ../../data/ -split_dir ../../data/split -save_dir ../../results/RVTrans
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
sys.path.append('../')
from utils.model_quantitative import RVTrans
from utils.dataloader import load_matrix_data



def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    ## Training
    parser.add_argument('-lr', default=1e-4, help='Learning rate')
    parser.add_argument('-batch_size', default=64, help='Training batch size')
    parser.add_argument('-nepoch', default=30, help='Number of epochs')
    parser.add_argument('-seed', default=42, help='Number of epochs')
    ## model
    parser.add_argument('-accumulation_step', default=None, help='Accumulation step, default to loop across [8, 32]')
    parser.add_argument('-depth', default=None, help='Network depth, default to loop across [6, 12, 18]')
    parser.add_argument('-patch_size', default=None, help='patch_size, default to loop across [128, 64, 32, 16]')
    parser.add_argument('-output_dim', default=1)
    parser.add_argument('-embed_dim', default=512, help='embed_dim')
    parser.add_argument('-emb_dim1', default=256, help='emb_dim1')
    parser.add_argument('-num_heads', default=8, help='Number of head')
    parser.add_argument('-input_drop', default=0.25, help='Input dropout')
    parser.add_argument('-pred_drop', default=0.5, help='Prediction layer dropout')
    ## hl-gauss参数
    parser.add_argument('-num_bins', default=100, help='')
    parser.add_argument('-sigma_to_bin_ratio', default=2., help='')
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
    parser.add_argument('-renormalize', default=False)
    parser.add_argument('-zscore_features', default=False)
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
depths = [6, 12, 18] if args.depth is None else [args.depth]
patch_sizes = [128, 64, 32, 16] if args.patch_size is None else [args.patch_size]

output_dim = int(args.output_dim)
embed_dim = int(args.embed_dim)
emb_dim1 = int(args.emb_dim1)
num_heads = int(args.num_heads)
input_drop = float(args.input_drop)
pred_drop = float(args.pred_drop)
## data
data_dir = str(args.data_dir)
cutoff = float(args.cutoff) 
pheno_name = str(args.pheno_name)
rare_gene = str(args.rare_gene)
split_dir = str(args.split_dir)
z_thre = float(args.z_thre)
## hl-gauss
num_bins = int(args.num_bins)
sigma_to_bin_ratio = float(args.sigma_to_bin_ratio)
## others
save_dir = str(args.save_dir)
renormalize = bool(args.renormalize)
zscore_features = bool(args.zscore_features)


save_name = f'Rare_{rare_gene}_cutoff{cutoff}_{pheno_name}_{z_thre}_{lr}_{nepoch}_{embed_dim}_{num_heads}'
os.system('mkdir -p ' + f"{save_dir}/results/{rare_gene}/{save_name}")
os.system('mkdir -p ' + f'{save_dir}/models/{rare_gene}/{save_name}')
os.system('mkdir -p ' + f'{save_dir}/PRS/{rare_gene}/{pheno_name}')
# Init tensorboard
os.system('mkdir -p ' + f'{save_dir}/models/{rare_gene}/{save_name}/runs')



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
# print(z_value_features[indx_resort])

if zscore_features:
    x_mean = np.mean(x_train_for_dataloader, axis=0)
    x_std= np.std(x_train_for_dataloader, axis=0)
    x_train_for_dataloader = (x_train_for_dataloader - x_mean)/x_std
    x_val_for_dataloader = (x_train_for_dataloader - x_mean)/x_std
    x_test_for_dataloader = (x_test_for_dataloader - x_mean)/x_std



train_dataset = load_matrix_data(x = x_train_for_dataloader, y=y_train, cov_fea=None)
ukb_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
print(f"train loader length: {len(ukb_loader)}")
niter = len(ukb_loader)
val_dataset = load_matrix_data(x = x_val_for_dataloader, y=y_val, cov_fea=None)
ukb_loader_val = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
print(f"val loader length: {len(ukb_loader_val)}")
test_dataset = load_matrix_data(x = x_test_for_dataloader, y=y_test, cov_fea=None)
ukb_loader_test = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)
print(f"test loader length: {len(ukb_loader_test)}")
print(f'Number of features = {x_train_for_dataloader.shape[1]}')

num_embeddings = int(max(np.max(x_train_for_dataloader), np.max(x_val_for_dataloader), np.max(x_test_for_dataloader))) + 1
print(f"frequency embedding length: {num_embeddings}")
pheno_max = max(torch.max(y_train), torch.max(y_val_test))
pheno_min = min(torch.min(y_train), torch.min(y_val_test))
print(f'Range of phenotype = [{pheno_min}, {pheno_max}]')



params = []
for i in range(0,len(accumulation_steps)):
    for k in range(0,len(depths)):  
        for j in range(0,len(patch_sizes)):
            params.append([accumulation_steps[i], patch_sizes[j], depths[k]])


for jjj in range(0, len(params)):

    save_ex = f"{params[jjj][0]}_{params[jjj][1]}_{params[jjj][2]}_{num_bins}_{sigma_to_bin_ratio}"
    accumulation_step = params[jjj][0]
    print(params[jjj])
    if x_train_for_dataloader.shape[1] <= params[jjj][1]:
        continue

    ## Init tensorboard
    os.system('mkdir -p ' + f'{save_dir}/models/{rare_gene}/{save_name}/runs/{save_ex}')
    writer = SummaryWriter(log_dir=f'{save_dir}/models/{rare_gene}/{save_name}/runs/{save_ex}')
    

    ## Load model
    nnet = RVTrans(input_size=x_train_for_dataloader.shape[1], patch_size=params[jjj][1], in_chans = 1, out_chans=1, embed_dim=embed_dim, 
            depth=params[jjj][2], num_heads=num_heads,mlp_ratio=4.,qkv_bias=False, qk_scale=None, 
            norm_layer=torch.nn.LayerNorm, mlp_time_embed=False, 
            num_embeddings=num_embeddings, embed_dim1=emb_dim1, # embedding
            min_value = pheno_min, max_value = pheno_max, num_bins = num_bins, sigma_to_bin_ratio = sigma_to_bin_ratio, # hl-gauss
            use_checkpoint=False, conv=True, skip=True, 
            attn_drop=0.1,
            proj_drop=0.1, 
            pred_drop=pred_drop,
            out_class=output_dim) 


    nparam = 0
    for p in nnet.parameters():
        if p.requires_grad is True:
            nparam = nparam + np.prod(p.shape)
    print('Number of param = ' + str(nparam / 1000000) + 'M')

    nnet.train()
    nnet.to(device)


    ## Load loss
    cls_num_list, bin_edges = train_dataset.get_cls_num_list(num_bins)
    beta = 0.9  ## 0.9999

    params11 = ([p for p in nnet.parameters()])
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
            if torch.sum(_batch[1])==0:
                continue

            ## renormalize data
            if renormalize:
                data_mean = torch.mean(data_in, dim=2, keepdim=True)
                data_std = torch.std(data_in, dim=2, keepdim=True)
                data_in = (data_in - data_mean)/(data_std+1e-5)
           
            context = data_in
            pred = nnet(context)

            loss = nnet(context, labels.squeeze())/accumulation_step

            loss.backward()
            torch.nn.utils.clip_grad_norm_(nnet.parameters(), 1.0)

            if (batch_idx+1) % accumulation_step == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                optimizer.zero_grad()
                lr_scheduler.step()
                # ema(nnet_ema, nnet, 0.999)

            if batch_idx % 2000 == 0:
                loss_print = loss.detach().item()
                print('Iter: ' + str(batch_idx) + ', Loss:' + str(loss_print))


        writer.add_scalar('Loss/train', loss.detach().item(), ep)
        lr_cur = optimizer.param_groups[0]['lr']
        writer.add_scalar('lr/train', lr_cur, ep)
        
        nnet.eval() 
        if ep == (nepoch-1):
            torch.save(nnet.state_dict(), f'{save_dir}/models/{rare_gene}/{save_name}/' + 'model_ep' + str(ep) + f"_{save_ex}" + '.pth')
        

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

                pred = nnet(data_in)
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
                best_model = copy.deepcopy(nnet)
                best_ep = ep
                val_pred_best = pred_all
                train_pred_best = pred_all_train


        #do test for current epoch
        with torch.no_grad():

            pred_all = np.zeros((len(y_test), y_train.shape[1]))
            for batch_idx, _batch in enumerate(ukb_loader_test):
                data_in = _batch[0].to(device)
                ## renormalize data
                if renormalize:
                    data_mean = torch.mean(data_in, dim=2, keepdim=True)
                    data_std = torch.std(data_in, dim=2, keepdim=True)
                    data_in = (data_in - data_mean)/(data_std+1e-5)

                pred = nnet(data_in)
                pred_all[_batch[-1].cpu().numpy(),:] = pred.detach().cpu().numpy().reshape(-1,1)

                if batch_idx==0:
                    id_index = _batch[2].numpy()
                else:
                    id_index = np.concatenate((id_index, _batch[2].numpy()))
            
            test_r = np.corrcoef(pred_all.flatten(), y_test.numpy().flatten())[0,1] 
            writer.add_scalar('R/test', test_r, ep)
            print(f'Epoch: {ep}, test r = {test_r}')
            test_r_ls[ep] = test_r

        nnet.train() 


    np.save(f'{save_dir}/results/{rare_gene}/{save_name}/mlp_{save_ex}_val_r.npy', val_r_ls)
    np.save(f'{save_dir}/results/{rare_gene}/{save_name}/mlp_{save_ex}_test_r.npy', test_r_ls)
    torch.save(best_model.state_dict(), f'{save_dir}/models/{rare_gene}/{save_name}/' + 'model_best_ep' + str(best_ep) + f"_{save_ex}" + '.pth')

       

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
            
        pred_all_label = (pred_all >= 0.5).astype(int)

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
    res_df.to_csv(f"{save_dir}/PRS/{rare_gene}/{pheno_name}/RVTrans_PRS_{save_ex}_cutoff{cutoff}.txt", sep="\t", index=False)
