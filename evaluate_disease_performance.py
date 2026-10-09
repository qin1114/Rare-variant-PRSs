"""
python evaluate_disease_performance.py -model_name LightGBM -cutoff 0.01 -rare_gene Coding -cov_file ../../data/Biochemistry_matched_data.csv -pheno_file ../../data/pheno_ls_5%.txt -save_dir ../../results/ -model_file ../../results -split_dir ../../data/split -cvprs_file ../../results/PRS-CS_new/Binary
"""
import pandas as pd
import os, argparse, sys, math
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append('../')
from utils.evaluation_matrix_binary import calculate_liability_r2, calculate_OR_value, calculate_IDI_value


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    ## Training
    parser.add_argument('-model_name', default='LightGBM', required=True, help='Model name')
    parser.add_argument('-cutoff', default=0.01, required=True, help='MAF threshold for burden score')
    parser.add_argument('-rare_gene', default="Coding", required=True, choices=["Coding", "Noncoding"], help='input type of rare gene burden score')
    parser.add_argument('-cov_file', default=None, required=True, help='filepath of covariates')
    parser.add_argument('-pheno_file', default=None, required=True, help='filepath of list of phenotype')
    parser.add_argument('-save_dir', default=None, required=True, help='output folder')
    parser.add_argument('-model_file', default=None, required=True, help='folder of rvPRS')
    parser.add_argument('-split_dir', default=None, required=True, help='folder of data split')
    parser.add_argument('-cvprs_file', default=None, required=True, help='folder of cvPRS')
    return parser

parser = cli_parser()
args = parser.parse_args()

cutoff = float(args.cutoff)
model_name = str(args.model_name)
rare_gene = str(args.rare_gene)
cov_file = str(args.cov_file)
pheno_file = str(args.pheno_file)
save_dir = str(args.save_dir)
model_file = str(args.model_file)
split_dir = str(args.split_dir)
cvprs_file = str(args.cvprs_file)


pheno_df = pd.read_csv(pheno_file, header=None)
pheno_ls = pheno_df[0].values.tolist()

bioche_data = pd.read_csv(cov_file, sep="\t")
cov_ls = ["21003-0.0", "31-0.0"] + [f"22009-0.{pc}" for pc in range(1,11)]
cov_df = bioche_data[["eid"]+cov_ls]
cov_df["IID"] = cov_df["eid"].astype(str) + "_" + cov_df["eid"].astype(str)


for model in [model_name]:
    for cutoff in [cutoff]:

        res_df = pd.DataFrame([])
        for pheno_name in pheno_ls:
            
            ## load rare PRS
            if model == 'RVTrans':
                ## select best params first
                patch_sizes = [128, 64, 32, 16]
                depths = [6, 12, 18]
                accumulation_steps = [8, 32]
                best_auc = -1
                for i in range(0,len(accumulation_steps)):
                    for j in range(0,len(patch_sizes)):
                        for k in range(0,len(depths)):
                            rs_path = f"{model_file}/{model}/PRS/{rare_gene}/{pheno_name}/RVTrans_PRS_{accumulation_steps[i]}_{patch_sizes[j]}_{depths[k]}_cutoff{cutoff}.txt"
                            if os.path.exists(rs_path):
                                rvprs_df_cur = pd.read_csv(rs_path, sep='\t')
                            else:
                                continue
                            rvprs_df_val_cur = rvprs_df_cur[rvprs_df_cur['dataset']=='val']
                            y_val = rvprs_df_val_cur["disease"].values
                            val_pred = rvprs_df_val_cur["Probability"].values
                            auc_val = roc_auc_score(y_val, val_pred)
                            if auc_val >= best_auc:
                                best_auc = auc_val
                                model_df_init = rvprs_df_cur

            elif model == 'MLP':
                ## select best params first
                accumulation_steps = [8, 32]
                hidden_dim = [128, 256, 512]
                best_auc = -1
                for i in range(0,len(accumulation_steps)):
                    for j in range(0,len(hidden_dim)):
                        rs_path = f"{model_file}/{model}/PRS/{rare_gene}/{pheno_name}/MLP_PRS_{accumulation_steps[i]}_{hidden_dim[j]}_cutoff{cutoff}.txt"
                        if os.path.exists(rs_path):
                            rvprs_df_cur = pd.read_csv(rs_path, sep='\t')
                        else:
                            continue
                        rvprs_df_val_cur = rvprs_df_cur[rvprs_df_cur['dataset']=='val']
                        y_val = rvprs_df_val_cur["disease"].values
                        val_pred = rvprs_df_val_cur["Probability"].values
                        auc_val = roc_auc_score(y_val, val_pred)
                        if auc_val >= best_auc:
                            best_auc = auc_val
                            model_df_init = rvprs_df_cur

            else:                
                model_df_init = pd.read_csv(f"{model_file}/{model}/PRS/{rare_gene}/{model}_PRS_{pheno_name}_cutoff{cutoff}.txt", sep="\t")
            

            model_df_init = pd.merge(model_df_init, cov_df, on="IID")
            model_df_train = model_df_init[model_df_init["dataset"]=="train"]

            data_split = pd.read_csv(f"{split_dir}/{pheno_name}/data_split.csv")
            data_split['IID'] = data_split["eid"].astype(str) + '_' + data_split["eid"].astype(str)
            model_df_init = pd.merge(model_df_init, data_split[['IID', 'split']], on='IID')
            model_df = model_df_init[model_df_init[f"split"]=="test"]
            model_val_df = model_df_init[model_df_init[f"split"]=="val"]


            # load common PRS
            common_df = pd.read_csv(f"{cvprs_file}/{pheno_name}/PRS-CS_PRS_test.txt", sep="\t")
            common_df = common_df[["IID", "SCORE"]]
            common_df.columns = ["IID", "Common_PRS"]
            final_data = pd.merge(model_df, common_df[["IID", "Common_PRS"]], on="IID")
            
            
            # ========== R ==========
            y_val = model_val_df["disease"].values
            rare_val_pred = model_val_df["Probability"].values
            K = np.mean(model_df_init['disease'].values)

            auc_val = roc_auc_score(y_val, rare_val_pred)
            try:
                r2_rare = calculate_liability_r2(final_data, use_common=True, K=K)["R2_lia"]
                if r2_rare<0:
                    r_rare = 0
                else:
                    r_rare = math.sqrt(r2_rare)
            except:
                r_rare = -1

            try:
                r_common = calculate_liability_r2(final_data, use_common=False, score_name='Common_PRS', K=K)["R2_lia"]
                if r_common<0:
                    r_common = 0
                else:
                    r_common = math.sqrt(r_common)
            except:
                r_common = -1
            

            # ========== odds_ratio ==========
            OR_rare, (CI_lower_rare, CI_upper_rare), p_value_rare, OR_common, (CI_lower_common, CI_upper_common), p_value_common = calculate_OR_value(res_df=final_data)
            OR2_rare, (OR2_CI_lower_rare, OR2_CI_upper_rare), OR2_p_value_rare, OR2_common, (OR2_CI_lower_common, OR2_CI_upper_common), OR2_p_value_common = calculate_OR_value(res_df=final_data, OR_thre=2) 
            OR5_rare, (OR5_CI_lower_rare, OR5_CI_upper_rare), OR5_p_value_rare, OR5_common, (OR5_CI_lower_common, OR5_CI_upper_common), OR5_p_value_common = calculate_OR_value(res_df=final_data, OR_thre=5) 


            # ========== IDI ==========
            IDI_res_rare = calculate_IDI_value(res_df=final_data, use_common=False)
            IDI_rare = IDI_res_rare["IDI"]
            IDI_P_rare = IDI_res_rare["IDI_p_value"]
            IDI_CI_rare = f'[{IDI_res_rare["IDI_CI_lower"]}, {IDI_res_rare["IDI_CI_upper"]}]'

            IDI_res_common = calculate_IDI_value(res_df=final_data, use_common=True)
            IDI_common = IDI_res_common["IDI"]
            IDI_P_common = IDI_res_common["IDI_p_value"]
            IDI_CI_common= f'[{IDI_res_common["IDI_CI_lower"]}, {IDI_res_common["IDI_CI_upper"]}]'


            cur_df = pd.DataFrame({"pheno_name": [pheno_name], "cutoff": [cutoff], "model": model, "data": rare_gene, 'val_auc': [auc_val], "R_rare": [r_rare], "R_common": [r_common], 
                    "IDI_rare": [IDI_rare], "IDI_CI_rare": [IDI_CI_rare], "IDI_p_value_rare": [IDI_P_rare], "IDI_common": [IDI_common], "IDI_CI_common": [IDI_CI_common],"IDI_p_value_common": [IDI_P_common], 
                    "OR_rare": [OR_rare], "OR_CI_rare": f'({CI_lower_rare}, {CI_upper_rare})', 'OR_p_value_rare': [p_value_rare], "OR_common": [OR_common],
                    "OR2_rare": [OR2_rare], "OR2_CI_rare": f'({OR2_CI_lower_rare}, {OR2_CI_upper_rare})', 'OR2_p_value_rare': [OR2_p_value_rare], "OR2_common": [OR2_common], 
                    "OR5_rare": [OR5_rare], "OR5_CI_rare": f'({OR5_CI_lower_rare}, {OR5_CI_upper_rare})', 'OR5_p_value_rare': [OR5_p_value_rare], "OR5_common": [OR5_common]})
            print(cur_df[["pheno_name", 'val_auc', 'R_rare', 'OR_rare', 'IDI_rare']])
            res_df = pd.concat([res_df, cur_df], axis=0)


        res_df.to_csv(
            f'{save_dir}/Binary_{rare_gene}_{model}_cutoff{cutoff}.csv',    
            sep='\t',       
            index=False,      
            header=True,   
            quoting=3,       
            escapechar='\\'     
        )
        print(res_df.describe())


