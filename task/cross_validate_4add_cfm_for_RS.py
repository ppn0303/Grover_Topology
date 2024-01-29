"""
The cross validation function for finetuning.
This implementation is adapted from
https://github.com/chemprop/chemprop/blob/master/chemprop/train/cross_validate.py
"""
import os
import time
from argparse import Namespace
from logging import Logger
from typing import Tuple

import numpy as np

from grover.util.utils import get_task_names
from grover.util.utils import makedirs
from task.run_evaluation import run_evaluation, run_evaluation_cfm
from task.train import run_training, run_training_cfm

import random
import torch

def cross_validate(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """
    k-fold cross validation.

    :return: A tuple of mean_score and std_score.
    """
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)

    # Run training with different random seeds for each fold
    all_scores = []
    time_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        if args.parser_name == "finetune":
            model_scores = run_training(args, time_start, logger)
        else:
            model_scores = run_evaluation(args, logger)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    # Report scores for each fold
    info(f'{args.num_folds}-fold cross validation')

    for fold_num, scores in enumerate(all_scores):
        info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(task_names, scores):
                info(f'Seed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')

    # Report scores across models
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'overall_{args.split_type}_test_{args.metric}={mean_score:.6f}')
    info(f'std={std_score:.6f}')

    if args.show_individual_scores:
        for task_num, task_name in enumerate(task_names):
            info(f'Overall test {task_name} {args.metric} = '
                 f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    return mean_score, std_score

def setup(seed):
    # frozen random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def randomsearch(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """
    randomsearch

    :return: A tuple of mean_score and std_score.
    """
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)

    #randomize parameter list
    max_lr_list = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]#, 0.0009, 0.001]
    lr_rate=[2,3,4,5,6,7,8,9,10]
    dropout_list = [0, 0.05, 0.1, 0.15, 0.2]
    attn_hidden_list = 128
    attn_out_list = [4, 8]
    dist_coff_list = [0.05, 0.1, 0.15]
    bond_drop_rate_list = [0, 0.2, 0.4, 0.6]
    ffn_num_layers_list = [2, 3]
    ffn_num_layers_list = [2, 3, 4, 5]
    ffn_dense_list = [300, 500, 700, 900, 1100, 1300]
    smote_rate_list = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # Run training with different random seeds for each fold
    all_scores_main_metric = []
    all_scores = []
    params = []
    time_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    for iter_num in range(0, args.n_iters):
        info(f'iter {iter_num}')

        #randomize parameter
        np.random.seed()
        random.seed()
        args.init_lr = args.max_lr / 10
        args.max_lr = np.random.choice(max_lr_list, 1)[0]
        args.final_lr = args.max_lr / np.random.choice(lr_rate, 1)[0]
        args.dropout = np.random.choice(dropout_list, 1)[0]
        args.attn_out = np.random.choice(attn_out_list, 1)[0]
        args.dist_coff = np.random.choice(dist_coff_list, 1)[0]
        args.bond_drop_rate = np.random.choice(bond_drop_rate_list, 1)[0]
        args.ffn_num_layers = np.random.choice(ffn_num_layers_list, 1)[0]
        args.ffn_hidden_size = np.random.choice(ffn_dense_list, 1)[0]
        if args.smote==True : 
            args.smote_rate = np.random.choice(smote_rate_list, 1)[0]
            params.append(f'\n{iter_num}th search parameter : init_lr is {args.init_lr} \n final_lr rate is {args.final_lr} \n dropout is {args.dropout} \n attn_out is {args.attn_out} \n dist_coff is {args.dist_coff} \n bond_drop_rate is {args.bond_drop_rate} \n ffn_num_layers is {args.ffn_num_layers} \n ffn_hidden_size is {args.ffn_hidden_size} \n batch_size is {args.batch_size} \n smote_rate is {args.smote_rate}')
        else : 
            params.append(f'\n{iter_num}th search parameter : init_lr is {args.init_lr} \n final_lr rate is {args.final_lr} \n dropout is {args.dropout} \n attn_out is {args.attn_out} \n dist_coff is {args.dist_coff} \n bond_drop_rate is {args.bond_drop_rate} \n ffn_num_layers is {args.ffn_num_layers} \n ffn_hidden_size is {args.ffn_hidden_size} \n batch_size is {args.batch_size}')
        info(params[iter_num])

        args.seed = init_seed                        # if change this, result will be change
        iter_dir = os.path.join(save_dir, f'iter_{iter_num}')
        args.save_dir = iter_dir
        makedirs(args.save_dir)

        fold_scores = []
        if args.confusionmatrix:
            scores_AUC = []
            scores_ACC = []
            scores_REC = []
            scores_PREC = []
            scores_SPEC = []
            scores_F1 = []
            scores_BA = []
            scores_TP = []
            scores_FP = []
            scores_TN = []
            scores_FN = []
        for fold_num in range(args.num_folds):
            info(f'Fold {fold_num}')
            args.seed = init_seed + fold_num
            args.save_dir = os.path.join(iter_dir, f'fold_{fold_num}')
            makedirs(args.save_dir)
            if args.parser_name == "finetune" and args.confusionmatrix:
                model_scores, AUC, ACC, REC, PREC, SPEC, F1, BA, TP, FP, TN, FN = run_training_cfm(args, time_start, logger)
                scores_AUC.append(AUC)
                scores_ACC.append(ACC)
                scores_REC.append(REC)
                scores_PREC.append(PREC)
                scores_SPEC.append(SPEC)
                scores_F1.append(F1)
                scores_BA.append(BA)
                scores_TP.append(TP)
                scores_FP.append(FP)
                scores_TN.append(TN)
                scores_FN.append(FN)
            elif args.parser_name == "finetune":
                model_scores = run_training(args, time_start, logger)
            else:
                model_scores = run_evaluation(args, logger)
                
            #change below line for compare average score
            fold_scores.append(model_scores)
        
        fold_scores = np.array(fold_scores)

        # Report scores for each fold
        info(f'\n{args.num_folds}-fold validation')
        info(f'{params[iter_num]}\n')

        for fold_num, scores in enumerate(fold_scores):
            info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

            if args.show_individual_scores:
                for task_name, score in zip(task_names, scores):
                    info(f'Seed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')

        # Report scores across models
        fold_avg_scores = np.nanmean(fold_scores, axis=1)  # average score for each model across tasks
        fold_mean_score, fold_std_score = np.nanmean(fold_avg_scores), np.nanstd(fold_avg_scores)
        info(f'overall_{args.split_type}_test_{args.metric}={fold_mean_score:.6f}')
        info(f'std={fold_std_score:.6f}\n')

        if args.show_individual_scores:
            for task_num, task_name in enumerate(task_names):
                info(f'Overall test {task_name} {args.metric} = '
                     f'{np.nanmean(fold_scores[:, task_num]):.6f} +/- {np.nanstd(fold_scores[:, task_num]):.6f}')

        all_scores_main_metric.append(fold_mean_score)
        
        if args.confusionmatrix:
            scores_AUC = np.array(scores_AUC)
            scores_ACC = np.array(scores_ACC)
            scores_REC = np.array(scores_REC)
            scores_PREC = np.array(scores_PREC)
            scores_SPEC = np.array(scores_SPEC)
            scores_F1 = np.array(scores_F1)
            scores_BA = np.array(scores_BA)
            scores_TN = np.array(scores_TN)
            scores_FN = np.array(scores_FN)
            scores_TP = np.array(scores_TP)
            scores_FP = np.array(scores_FP)
            # Report scores for each fold
            info(f'{args.num_folds}-fold cross validation')

            # Report scores across models
            avg_scores_AUC = np.nanmean(scores_AUC, axis=1)  # average score for each model across tasks
            mean_score_AUC, std_score_AUC = np.nanmean(avg_scores_AUC), np.nanstd(avg_scores_AUC)
            info(f'overall_{args.split_type}_test_AUC={mean_score_AUC:.6f}')
            info(f'std={std_score_AUC:.6f}')

            avg_scores_ACC = np.nanmean(scores_ACC, axis=1)  # average score for each model across tasks
            mean_score_ACC, std_score_ACC = np.nanmean(avg_scores_ACC), np.nanstd(avg_scores_ACC)
            info(f'overall_{args.split_type}_test_Accuracy={mean_score_ACC:.6f}')
            info(f'std={std_score_ACC:.6f}')

            avg_scores_REC = np.nanmean(scores_REC, axis=1)  # average score for each model across tasks
            mean_score_REC, std_score_REC = np.nanmean(avg_scores_REC), np.nanstd(avg_scores_REC)
            info(f'overall_{args.split_type}_test_Recall={mean_score_REC:.6f}')
            info(f'std={std_score_REC:.6f}')

            avg_scores_PREC = np.nanmean(scores_PREC, axis=1)  # average score for each model across tasks
            mean_score_PREC, std_score_PREC = np.nanmean(avg_scores_PREC), np.nanstd(avg_scores_PREC)
            info(f'overall_{args.split_type}_test_Precision={mean_score_PREC:.6f}')
            info(f'std={std_score_PREC:.6f}')

            avg_scores_SPEC = np.nanmean(scores_SPEC, axis=1)  # average score for each model across tasks
            mean_score_SPEC, std_score_SPEC = np.nanmean(avg_scores_SPEC), np.nanstd(avg_scores_SPEC)
            info(f'overall_{args.split_type}_test_Specificity={mean_score_SPEC:.6f}')
            info(f'std={std_score_SPEC:.6f}')

            avg_scores_F1 = np.nanmean(scores_F1, axis=1)  # average score for each model across tasks
            mean_score_F1, std_score_F1 = np.nanmean(avg_scores_F1), np.nanstd(avg_scores_F1)
            info(f'overall_{args.split_type}_test_F1={mean_score_F1:.6f}')
            info(f'std={std_score_F1:.6f}')

            avg_scores_BA = np.nanmean(scores_BA, axis=1)  # average score for each model across tasks
            mean_score_BA, std_score_BA = np.nanmean(avg_scores_BA), np.nanstd(avg_scores_BA)
            info(f'overall_{args.split_type}_test_BA={mean_score_BA:.6f}')
            info(f'std={std_score_BA:.6f}')

            avg_scores_TP = np.nanmean(scores_TP)  # average score for each model across tasks
            mean_score_TP, std_score_TP = np.nanmean(avg_scores_TP), np.nanstd(avg_scores_TP)

            avg_scores_FP = np.nanmean(scores_FP)  # average score for each model across tasks
            mean_score_FP, std_score_FP = np.nanmean(avg_scores_FP), np.nanstd(avg_scores_FP)

            avg_scores_TN = np.nanmean(scores_TN)  # average score for each model across tasks
            mean_score_TN, std_score_TN = np.nanmean(avg_scores_TN), np.nanstd(avg_scores_TN)

            avg_scores_FN = np.nanmean(scores_FN)  # average score for each model across tasks
            mean_score_FN, std_score_FN = np.nanmean(avg_scores_FN), np.nanstd(avg_scores_FN)
            info(f'TP : {mean_score_TP:.6f}\tFP : {mean_score_FP:.6f}')
            info(f'FN : {mean_score_FN:.6f}\tTN : {mean_score_TN:.6f}')
            
            all_scores.append([mean_score_AUC, mean_score_ACC, mean_score_REC, mean_score_PREC, mean_score_SPEC, mean_score_F1, mean_score_BA, mean_score_TP, mean_score_FP, mean_score_TN, mean_score_FN])


            if args.show_individual_scores:
                for task_num, task_name in enumerate(task_names):
                    info(f'Overall test {task_name} {args.metric} = '
                         f'{np.nanmean(all_scores_main_metric[:, task_num]):.6f} +/- {np.nanstd(all_scores_main_metric[:, task_num]):.6f}')

############fold end, save fold_data and initialize seed

        # best setting save
        if args.dataset_type=='classification' and args.confusionmatrix:
            if max(all_scores_main_metric)==fold_mean_score : 
                best_iter = iter_num
                best_score = fold_mean_score
                best_param = params[iter_num]
        elif args.dataset_type=='classification' : 
            if max(all_scores_main_metric)==fold_mean_score : 
                best_iter = iter_num
                best_score = fold_mean_score
                best_param = params[iter_num]
        else : 
            if min(all_scores_main_metric)==fold_mean_score : 
                best_iter = iter_num
                best_score = fold_mean_score
                best_param = params[iter_num]
############iter end

    all_scores_main_metric = np.array(all_scores_main_metric)

    # Report scores for each iter
    info(f'\n---- {args.n_iters}-iter random search ----')

    if args.confusionmatrix:
        all_scores = np.array(all_scores)
        for iter_num, scores in enumerate(all_scores):
            info(params[iter_num])
            info(f'Seed {init_seed} ==> test AUC = {np.nanmean(scores[0]):.6f}\n')
            info(f'Seed {init_seed} ==> test ACC = {np.nanmean(scores[1]):.6f}\n')
            info(f'Seed {init_seed} ==> test REC = {np.nanmean(scores[2]):.6f}\n')
            info(f'Seed {init_seed} ==> test PREC = {np.nanmean(scores[3]):.6f}\n')
            info(f'Seed {init_seed} ==> test SPEC = {np.nanmean(scores[4]):.6f}\n')
            info(f'Seed {init_seed} ==> test F1 = {np.nanmean(scores[5]):.6f}\n')
            info(f'Seed {init_seed} ==> test BA = {np.nanmean(scores[6]):.6f}\n')
            info(f'TP : {np.nanmean(scores[7]):.6f}\tFP : {np.nanmean(scores[8]):.6f}')
            info(f'FN : {np.nanmean(scores[10]):.6f}\tTN : {np.nanmean(scores[9]):.6f}')
    else:
        for iter_num, scores in enumerate(all_scores_main_metric):
            info(params[iter_num])
            info(f'Seed {init_seed} ==> test {args.metric} = {np.nanmean(scores):.6f}\n')

    # Report best model
    info(f'\nbest_iter : {best_iter}\nbest_score is {np.nanmean(best_score)}\nbest_param : {best_param}')

    return best_score

def gridsearch(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """
    gridsearch

    :return: A tuple of mean_score and std_score.
    """
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)


#Grid search parameter list    
    #max_lr_list = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]#, 0.0009, 0.001]
    max_lr_list = [0.0003]

    #lr_rate=[2,3,4,5,6,7,8,9,10]
    lr_rate_list=[10]

    #dropout_list = [0, 0.05, 0.1, 0.15, 0.2]
    dropout_list = [0.05]

    attn_hidden_list = 128
    attn_out_list = [4]

    #dist_coff_list = [0.05, 0.1, 0.15, 0.20]
    dist_coff_list = [0.05]

    #bond_drop_rate_list = [0, 0.2, 0.4, 0.6]
    bond_drop_rate_list = [0]

    ffn_num_layers_list = [3]
    ffn_dense_list = [300, 700, 1100, 1300]
    if args.smote==True : smote_rate_list = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    else : smote_rate_list = [1]


    # Run gridsearch with selected parameter
    all_scores = []
    params = []
    time_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    iter_num=0
    for ffnnum in range(len(ffn_num_layers_list)):
        for ffndense in range(len(ffn_dense_list)):
            for maxlr in range(len(max_lr_list)):
                for lrrate in range(len(lr_rate_list)):
                    for droprate in range(len(dropout_list)):
                        for bonddroprate in range(len(bond_drop_rate_list)):
                            for distcoff in range(len(dist_coff_list)):
                                for attnout in range(len(attn_out_list)):
                                    for smoterate in range(len(smote_rate_list)):
                                        info(f'iter {iter_num}')

                                        #select parameter
                                        args.ffn_hidden_size = ffn_dense_list[ffndense]
                                        args.ffn_num_layers = ffn_num_layers_list[ffnnum]
                                        args.max_lr = max_lr_list[maxlr]
                                        args.init_lr = args.max_lr / 10
                                        args.final_lr = args.max_lr / lr_rate_list[lrrate]
                                        if args.smote==True : 
                                            args.smote_rate = smote_rate_list[smoterate]
                                            params.append(f'\n{iter_num}th search parameter : init_lr is {args.init_lr} \n final_lr rate is {args.final_lr} \n dropout is {args.dropout} \n attn_out is {args.attn_out} \n dist_coff is {args.dist_coff} \n bond_drop_rate is {args.bond_drop_rate} \n ffn_num_layers is {args.ffn_num_layers} \n ffn_hidden_size is {args.ffn_hidden_size} \n batch_size is {args.batch_size} \n smote_rate is {args.smote_rate}')
                                        else : params.append(f'\n{iter_num}th search parameter : init_lr is {args.init_lr} \n final_lr rate is {args.final_lr} \n dropout is {args.dropout} \n attn_out is {args.attn_out} \n dist_coff is {args.dist_coff} \n bond_drop_rate is {args.bond_drop_rate} \n ffn_num_layers is {args.ffn_num_layers} \n ffn_hidden_size is {args.ffn_hidden_size} \n batch_size is {args.batch_size}')
                                        info(params[iter_num])

                                        args.seed = init_seed                        # if change this, result will be change
                                        iter_dir = os.path.join(save_dir, f'iter_{iter_num}')
                                        args.save_dir = iter_dir
                                        makedirs(args.save_dir)

                                        fold_scores = []
                                        for fold_num in range(args.num_folds):
                                            info(f'Fold {fold_num}')
                                            args.seed = init_seed + fold_num
                                            args.save_dir = os.path.join(iter_dir, f'fold_{fold_num}')
                                            makedirs(args.save_dir)
                                            if args.parser_name == "finetune":
                                                model_scores = run_training(args, time_start, logger)
                                            else:
                                                model_scores = run_evaluation(args, logger)
                                #change below line for compare average score
                                            fold_scores.append(model_scores)
                                        fold_scores = np.array(fold_scores)

                                        # Report scores for each fold
                                        info(f'\n{args.num_folds}-fold validation')
                                        info(f'{params[iter_num]}\n')

                                        for fold_num, scores in enumerate(fold_scores):
                                            info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

                                            if args.show_individual_scores:
                                                for task_name, score in zip(task_names, scores):
                                                    info(f'Seed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')

                                        # Report scores across models
                                        fold_avg_scores = np.nanmean(fold_scores, axis=1)  # average score for each model across tasks
                                        fold_mean_score, fold_std_score = np.nanmean(fold_avg_scores), np.nanstd(fold_avg_scores)
                                        info(f'overall_{args.split_type}_test_{args.metric}={fold_mean_score:.6f}')
                                        info(f'std={fold_std_score:.6f}\n')

                                        if args.show_individual_scores:
                                            for task_num, task_name in enumerate(task_names):
                                                info(f'Overall test {task_name} {args.metric} = '
                                                     f'{np.nanmean(fold_scores[:, task_num]):.6f} +/- {np.nanstd(fold_scores[:, task_num]):.6f}')

                                        all_scores.append(fold_mean_score)

                                ############fold end, save fold_data and initialize seed

                                        # best setting save
                                        if args.dataset_type=='classification' : 
                                            if max(all_scores)==fold_mean_score : 
                                                best_iter = iter_num
                                                best_score = fold_mean_score
                                                best_param = params[iter_num]
                                        else : 
                                            if min(all_scores)==fold_mean_score : 
                                                best_iter = iter_num
                                                best_score = fold_mean_score
                                                best_param = params[iter_num]
                                        iter_num+=1

############iter end

    all_scores = np.array(all_scores)

    # Report scores for each iter
    info(f'\n---- {args.n_iters}-iter random search ----')

    for iter_num, scores in enumerate(all_scores):
        info(params[iter_num])
        info(f'Seed {init_seed} ==> test {args.metric} = {np.nanmean(scores):.6f}\n')

    # Report best model
    info(f'\nbest_iter : {best_iter}\nbest_score is {np.nanmean(best_score)}\nbest_param : {best_param}')

    return best_score

"""
def gridsearch(args: Namespace, logger: Logger = None) -> Tuple[float, float]:

#    gridsearch

#    :return: A tuple of mean_score and std_score.

    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)

    # Grid search parameter list
    ffn_num_layers_list = [2, 3]
    ffn_dense_list = [300, 500, 700, 900, 1100, 1300]
#    ffn_dense_list = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300]
    max_lr_list = [0.0005]
    lr_rate=[10]
    smote_rate=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Run gridsearch with selected parameter
    all_scores = []
    params = []
    time_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    for iter_num in range(0, len(ffn_dense_list)):
        info(f'iter {iter_num}')

        #select parameter
        args.ffn_hidden_size = ffn_dense_list[iter_num]
        args.max_lr = np.random.choice(max_lr_list, 1)[0]
        args.init_lr = args.max_lr / 10
        args.final_lr = args.max_lr / np.random.choice(lr_rate, 1)[0]
        params.append(f'\n{iter_num}th search parameter : init_lr is {args.init_lr} \n final_lr rate is {args.final_lr} \n dropout is {args.dropout} \n attn_out is {args.attn_out} \n dist_coff is {args.dist_coff} \n bond_drop_rate is {args.bond_drop_rate} \n ffn_num_layers is {args.ffn_num_layers} \n ffn_hidden_size is {args.ffn_hidden_size} \n batch_size is {args.batch_size}')
        info(params[iter_num])

        args.seed = init_seed                        # if change this, result will be change
        iter_dir = os.path.join(save_dir, f'iter_{iter_num}')
        args.save_dir = iter_dir
        makedirs(args.save_dir)

        fold_scores = []
        for fold_num in range(args.num_folds):
            info(f'Fold {fold_num}')
            args.seed = init_seed + fold_num
            args.save_dir = os.path.join(iter_dir, f'fold_{fold_num}')
            makedirs(args.save_dir)
            if args.parser_name == "finetune":
                model_scores = run_training(args, time_start, logger)
            else:
                model_scores = run_evaluation(args, logger)
#change below line for compare average score
            fold_scores.append(model_scores)
        fold_scores = np.array(fold_scores)

        # Report scores for each fold
        info(f'\n{args.num_folds}-fold validation')
        info(f'{params[iter_num]}\n')

        for fold_num, scores in enumerate(fold_scores):
            info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

            if args.show_individual_scores:
                for task_name, score in zip(task_names, scores):
                    info(f'Seed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')

        # Report scores across models
        fold_avg_scores = np.nanmean(fold_scores, axis=1)  # average score for each model across tasks
        fold_mean_score, fold_std_score = np.nanmean(fold_avg_scores), np.nanstd(fold_avg_scores)
        info(f'overall_{args.split_type}_test_{args.metric}={fold_mean_score:.6f}')
        info(f'std={fold_std_score:.6f}\n')

        if args.show_individual_scores:
            for task_num, task_name in enumerate(task_names):
                info(f'Overall test {task_name} {args.metric} = '
                     f'{np.nanmean(fold_scores[:, task_num]):.6f} +/- {np.nanstd(fold_scores[:, task_num]):.6f}')

        all_scores.append(fold_mean_score)

############fold end, save fold_data and initialize seed

        # best setting save
        if args.dataset_type=='classification' : 
            if max(all_scores)==fold_mean_score : 
                best_iter = iter_num
                best_score = fold_mean_score
                best_param = params[iter_num]
        else : 
            if min(all_scores)==fold_mean_score : 
                best_iter = iter_num
                best_score = fold_mean_score
                best_param = params[iter_num]

############iter end

    all_scores = np.array(all_scores)

    # Report scores for each iter
    info(f'\n---- {args.n_iters}-iter random search ----')

    for iter_num, scores in enumerate(all_scores):
        info(params[iter_num])
        info(f'Seed {init_seed} ==> test {args.metric} = {np.nanmean(scores):.6f}\n')

    # Report best model
    info(f'\nbest_iter : {best_iter}\nbest_score is {np.nanmean(best_score)}\nbest_param : {best_param}')

    return best_score
"""

def make_confusion_matrix(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """
    make_confusion_matrix

    :return: A tuple of mean_score and std_score.
    """
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)

    # Run training with different random seeds for each fold
    scores_AUC = []
    scores_ACC = []
    scores_REC = []
    scores_PREC = []
    scores_SPEC = []
    scores_F1 = []
    scores_BA = []
    scores_TP = []
    scores_FP = []
    scores_TN = []
    scores_FN = []

    time_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    if args.dataset_type == "regression":
        for fold_num in range(args.num_folds):
            info(f'Fold {fold_num}')
            args.seed = init_seed + fold_num
            args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
            #makedirs(args.save_dir)
        
            model_scores = run_evaluation(args, logger)
            scores_AUC.append(model_scores)
        # Report scores for each fold
        info(f'{args.num_folds}-fold cross validation')

        # Report scores across models
        avg_scores = np.nanmean(scores_AUC, axis=1)  # average score for each model across tasks
        mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
        info(f'overall_{args.split_type}_test_AUC={mean_score:.6f}')
        info(f'std={std_score:.6f}')

        return mean_score, std_score

            
    else:
        for fold_num in range(args.num_folds):
            info(f'Fold {fold_num}')
            args.seed = init_seed + fold_num
            args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
            makedirs(args.save_dir)
            
            AUC, ACC, REC, PREC, SPEC, F1, BA, TP, FP, TN, FN = run_evaluation_cfm(args, logger)
            scores_AUC.append(AUC)
            scores_ACC.append(ACC)
            scores_REC.append(REC)
            scores_PREC.append(PREC)
            scores_SPEC.append(SPEC)
            scores_F1.append(F1)
            scores_BA.append(BA)
            scores_TP.append(TP)
            scores_FP.append(FP)
            scores_TN.append(TN)
            scores_FN.append(FN)
        scores_AUC = np.array(scores_AUC)
        scores_ACC = np.array(scores_ACC)
        scores_REC = np.array(scores_REC)
        scores_PREC = np.array(scores_PREC)
        scores_SPEC = np.array(scores_SPEC)
        scores_F1 = np.array(scores_F1)
        scores_BA = np.array(scores_BA)
        scores_TN = np.array(scores_TN)
        scores_FN = np.array(scores_FN)
        scores_TP = np.array(scores_TP)
        scores_FP = np.array(scores_FP)

        # Report scores for each fold
        info(f'{args.num_folds}-fold cross validation')

        # Report scores across models
        avg_scores_AUC = np.nanmean(scores_AUC, axis=1)  # average score for each model across tasks
        mean_score_AUC, std_score_AUC = np.nanmean(avg_scores_AUC), np.nanstd(avg_scores_AUC)
        info(f'overall_{args.split_type}_test_AUC={mean_score_AUC:.6f}')
        info(f'std={std_score_AUC:.6f}')

        avg_scores_ACC = np.nanmean(scores_ACC, axis=1)  # average score for each model across tasks
        mean_score_ACC, std_score_ACC = np.nanmean(avg_scores_ACC), np.nanstd(avg_scores_ACC)
        info(f'overall_{args.split_type}_test_Accuracy={mean_score_ACC:.6f}')
        info(f'std={std_score_ACC:.6f}')

        avg_scores_REC = np.nanmean(scores_REC, axis=1)  # average score for each model across tasks
        mean_score_REC, std_score_REC = np.nanmean(avg_scores_REC), np.nanstd(avg_scores_REC)
        info(f'overall_{args.split_type}_test_Recall={mean_score_REC:.6f}')
        info(f'std={std_score_REC:.6f}')

        avg_scores_PREC = np.nanmean(scores_PREC, axis=1)  # average score for each model across tasks
        mean_score_PREC, std_score_PREC = np.nanmean(avg_scores_PREC), np.nanstd(avg_scores_PREC)
        info(f'overall_{args.split_type}_test_Precision={mean_score_PREC:.6f}')
        info(f'std={std_score_PREC:.6f}')

        avg_scores_SPEC = np.nanmean(scores_SPEC, axis=1)  # average score for each model across tasks
        mean_score_SPEC, std_score_SPEC = np.nanmean(avg_scores_SPEC), np.nanstd(avg_scores_SPEC)
        info(f'overall_{args.split_type}_test_Specificity={mean_score_SPEC:.6f}')
        info(f'std={std_score_SPEC:.6f}')

        avg_scores_F1 = np.nanmean(scores_F1, axis=1)  # average score for each model across tasks
        mean_score_F1, std_score_F1 = np.nanmean(avg_scores_F1), np.nanstd(avg_scores_F1)
        info(f'overall_{args.split_type}_test_F1={mean_score_F1:.6f}')
        info(f'std={std_score_F1:.6f}')

        avg_scores_BA = np.nanmean(scores_BA, axis=1)  # average score for each model across tasks
        mean_score_BA, std_score_BA = np.nanmean(avg_scores_BA), np.nanstd(avg_scores_BA)
        info(f'overall_{args.split_type}_test_BA={mean_score_BA:.6f}')
        info(f'std={std_score_BA:.6f}')

        avg_scores_TP = np.nanmean(scores_TP)  # average score for each model across tasks
        mean_score_TP, std_score_TP = np.nanmean(avg_scores_TP), np.nanstd(avg_scores_TP)

        avg_scores_FP = np.nanmean(scores_FP)  # average score for each model across tasks
        mean_score_FP, std_score_FP = np.nanmean(avg_scores_FP), np.nanstd(avg_scores_FP)

        avg_scores_TN = np.nanmean(scores_TN)  # average score for each model across tasks
        mean_score_TN, std_score_TN = np.nanmean(avg_scores_TN), np.nanstd(avg_scores_TN)

        avg_scores_FN = np.nanmean(scores_FN)  # average score for each model across tasks
        mean_score_FN, std_score_FN = np.nanmean(avg_scores_FN), np.nanstd(avg_scores_FN)
        info(f'TP : {mean_score_TP:.6f}\tFP : {mean_score_FP:.6f}')
        info(f'FN : {mean_score_FN:.6f}\tTN : {mean_score_TN:.6f}')


        if args.show_individual_scores:
            for task_num, task_name in enumerate(task_names):
                info(f'Overall test {task_name} {args.metric} = '
                     f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

        return mean_score_AUC, std_score_AUC

