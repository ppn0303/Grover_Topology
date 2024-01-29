"""
The training function used in the finetuning task.
"""
import csv
import logging
import os
import pickle
import time
from argparse import Namespace
from logging import Logger
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from grover.data import MolCollator
from grover.data import StandardScaler
from grover.util.metrics import get_metric_func
from grover.util.nn_utils import initialize_weights, param_count
from grover.util.scheduler import NoamLR
from grover.util.utils import build_optimizer, build_lr_scheduler, makedirs, load_checkpoint, get_loss_func, \
    save_checkpoint, build_model
from grover.util.utils import get_class_sizes, get_data, split_data, get_task_names
from task.predict import predict, evaluate, evaluate_predictions, evaluate_predictions_cfm

import wandb

def train(epoch, model, data, loss_func, optimizer, scheduler,
          shared_dict, args: Namespace, n_iter: int = 0,
          logger: logging.Logger = None):
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: A MoleculeDataset (or a list of MoleculeDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    # debug = logger.debug if logger is not None else print

    model.train()

    # data.shuffle()

    loss_sum, iter_count = 0, 0
    cum_loss_sum, cum_iter_count = 0, 0


    mol_collator = MolCollator(shared_dict=shared_dict, args=args)

    num_workers = 4
    if type(data) == DataLoader:
        mol_loader = data
    else:
        mol_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=mol_collator)

    for _, item in enumerate(mol_loader):
        _, batch, features_batch, mask, targets = item
        if next(model.parameters()).is_cuda:
            mask, targets = mask.cuda(), targets.cuda()
        class_weights = torch.ones(targets.shape)

        if args.cuda:
            class_weights = class_weights.cuda()

        # Run model
        model.zero_grad()
        preds = model(batch, features_batch)
        loss = loss_func(preds, targets) * class_weights * mask
        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += args.batch_size

        cum_loss_sum += loss.item()
        cum_iter_count += 1

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += args.batch_size

        #if (n_iter // args.batch_size) % args.log_frequency == 0:
        #    lrs = scheduler.get_lr()
        #    loss_avg = loss_sum / iter_count
        #    loss_sum, iter_count = 0, 0
        #    lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))

    return n_iter, cum_loss_sum / cum_iter_count


def run_training(args: Namespace, time_start, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print


    # pin GPU to local rank.
    idx = args.gpu
    if args.gpu is not None:
        torch.cuda.set_device(idx)

    features_scaler, scaler, shared_dict, test_data, train_data, val_data = load_data(args, debug, logger)
    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.multi_class:
        sum_test_preds = np.zeros((len(test_smiles), args.multi_class_num))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            if len(args.checkpoint_paths) == 1:
                cur_model = 0
            else:
                cur_model = model_idx
            debug(f'Loading model {cur_model} from {args.checkpoint_paths[cur_model]}')
            model = load_checkpoint(args.checkpoint_paths[cur_model], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = build_model(model_idx=model_idx, args=args)

        if args.fine_tune_coff != 1 and args.checkpoint_paths is not None:
            debug("Fine tune fc layer with different lr")
            initialize_weights(model_idx=model_idx, model=model.ffn, distinct_init=args.distinct_init)

        # Get loss and metric functions
        loss_func = get_loss_func(args, model)

        optimizer = build_optimizer(model, args)

        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # set up wandb
        if args.wandb : 
            wandb.init(project=args.wandb_name)
            wandb.config = args
            wandb.watch(model)


        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Bulid data_loader
        shuffle = True
        mol_collator = MolCollator(shared_dict={}, args=args)
        train_data = DataLoader(train_data,
                                batch_size=args.batch_size,
                                shuffle=shuffle,
                                num_workers=10,
                                collate_fn=mol_collator)
        
        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        min_val_loss = float('inf')
        for epoch in range(args.epochs):
            s_time = time.time()
            n_iter, train_loss = train(
                epoch=epoch,
                model=model,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                shared_dict=shared_dict,
                logger=logger
            )
            t_time = time.time() - s_time
            s_time = time.time()
            val_scores, val_loss = evaluate(
                model=model,
                data=val_data,
                loss_func=loss_func,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                shared_dict=shared_dict,
                logger=logger,
                args=args
            )
            v_time = time.time() - s_time


            # Average validation score
            avg_val_score = np.nanmean(val_scores)


            # Logged after lr step
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.6f}'.format(train_loss),
                  'loss_val: {:.6f}'.format(val_loss),
                  f'{args.metric}_val: {avg_val_score:.4f}',
                  # 'auc_val: {:.4f}'.format(avg_val_score),
                  'cur_lr: {:.5f}'.format(scheduler.get_lr()[-1]),
                  't_time: {:.4f}s'.format(t_time),
                  'v_time: {:.4f}s'.format(v_time))

            if args.tensorboard:
                writer.add_scalar('loss/train', train_loss, epoch)
                writer.add_scalar('loss/val', val_loss, epoch)
                writer.add_scalar(f'{args.metric}_val', avg_val_score, epoch)

            if args.wandb :         
                wandb.log({"val_loss" : val_loss, "val_metrics" : val_scores})


            # Save model checkpoint if improved validation score
            if args.select_by_loss:
                if val_loss < min_val_loss:
                    min_val_loss, best_epoch = val_loss, epoch
                    save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)
            else:
                if args.minimize_score and avg_val_score < best_score or \
                        not args.minimize_score and avg_val_score > best_score:
                    best_score, best_epoch = avg_val_score, epoch
                    save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

            if epoch - best_epoch > args.early_stop_epoch:
                break

        ensemble_scores = 0.0

        # Evaluate on test set using model with best validation score
        if args.select_by_loss:
            info(f'Model {model_idx} best val loss = {min_val_loss:.6f} on epoch {best_epoch}')
        else:
            info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)

        test_preds, _ = predict(
            model=model,
            data=test_data,
            loss_func=loss_func,
            batch_size=args.batch_size,
            logger=logger,
            shared_dict=shared_dict,
            scaler=scaler,
            args=args
        )

        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            arg=args,
            logger=logger
        )

        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds, dtype=float)

        # Average test score
        avg_test_score = np.nanmean(test_scores)
        info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')

        if args.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(args.task_names, test_scores):
                info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')

        # Evaluate ensemble on test set
        avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

        ensemble_scores = evaluate_predictions(
            preds=avg_test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            arg=args,
            logger=logger
        )

        ind = [['preds'] * args.num_tasks + ['targets'] * args.num_tasks, args.task_names * 2]
        ind = pd.MultiIndex.from_tuples(list(zip(*ind)))
        if args.multi_class:
            data = np.concatenate([np.array([np.argmax(x) for x in avg_test_preds]).reshape(-1,1), np.array(test_targets)], 1)
        else:
            data = np.concatenate([np.array(avg_test_preds), np.array(test_targets)], 1)
        test_result = pd.DataFrame(data, index=test_smiles, columns=ind)
        test_result.to_csv(os.path.join(args.save_dir, 'test_result.csv'))

        # Average ensemble score
        avg_ensemble_test_score = np.nanmean(ensemble_scores)
        info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')

        # Individual ensemble scores
        if args.show_individual_scores:
            for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
                info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')

    return ensemble_scores



def load_data(args, debug, logger):
    """
    load the training data.
    :param args:
    :param debug:
    :param logger:
    :return:
    """
    # Get data
    debug('Loading data')
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    if data.data[0].features is not None:
        args.features_dim = len(data.data[0].features)
    else:
        args.features_dim = 0
    shared_dict = {}
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')
    # Split data
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args,
                             features_path=args.separate_test_features_path, logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args,
                            features_path=args.separate_val_features_path, logger=logger)
    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type,
                                              sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type,
                                             sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type,
                                                     sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)
    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    if args.smote == True:
        if args.dataset_type == 'classification':
            class_sizes = get_class_sizes(data)
            debug('Origin Class sizes')
            for i, task_class_sizes in enumerate(class_sizes):
                debug(f'{args.task_names[i]} '
                      f'{", ".join(f"{cls}: {int(size*args.train_data_size)}({size * 100:.2f}%)" for cls, size in enumerate(task_class_sizes))}')
        debug(f'Total size = {len(data):,} | '
              f'train size = {args.train_data_size:,} | val size = {len(val_data):,} | test size = {len(test_data):,}')
        args.train_data_size = len(train_data)
    
        debug('Smoted Class sizes')
        smoted_class_sizes = get_class_sizes(train_data)
        for i, task_class_sizes in enumerate(smoted_class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {int(size*args.train_data_size)}({size * 100:.2f}%)" for cls, size in enumerate(task_class_sizes))}')
        #note : there is some error of number because class_count is class_rate*data_length
        debug(f'Total size = {len(test_data)+len(train_data)+len(val_data):,} | '
              f'train size = {args.train_data_size:,} | val size = {len(val_data):,} | test size = {len(test_data):,}')
    else:
        if args.dataset_type == 'classification':
            class_sizes = get_class_sizes(data, args)
            debug('Class sizes')
            if not args.multi_class:
                for i, task_class_sizes in enumerate(class_sizes):
                    debug(f'{args.task_names[i]} '
                          f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')
            elif args.multi_class:
                for i in range(args.multi_class_num):
                    print(f'{i} : {class_sizes[i][0]:.2f}')
                    
        args.train_data_size = len(train_data)
        debug(f'Total size = {len(data):,} | '
              f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        _, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)

        val_targets = val_data.targets()
        scaled_val_targets = scaler.transform(val_targets).tolist()
        val_data.set_targets(scaled_val_targets)
    else:
        scaler = None
    return features_scaler, scaler, shared_dict, test_data, train_data, val_data


def save_splits(args, test_data, train_data, val_data):
    """
    Save the splits.
    :param args:
    :param test_data:
    :param train_data:
    :param val_data:
    :return:
    """
    with open(args.data_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        lines_by_smiles = {}
        indices_by_smiles = {}
        for i, line in enumerate(reader):
            smiles = line[0]
            lines_by_smiles[smiles] = line
            indices_by_smiles[smiles] = i

    all_split_indices = []
    for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
        with open(os.path.join(args.save_dir, name + '_smiles.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['smiles'])
            for smiles in dataset.smiles():
                writer.writerow([smiles])
        with open(os.path.join(args.save_dir, name + '_full.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for smiles in dataset.smiles():
                writer.writerow(lines_by_smiles[smiles])
        split_indices = []
        for smiles in dataset.smiles():
            split_indices.append(indices_by_smiles[smiles])
            split_indices = sorted(split_indices)
        all_split_indices.append(split_indices)
    with open(os.path.join(args.save_dir, 'split_indices.pckl'), 'wb') as f:
        pickle.dump(all_split_indices, f)
    return writer

#this is making emb_vector
def make_emb(epoch, model, data, loss_func, optimizer, scheduler,
          shared_dict, args: Namespace, n_iter: int = 0,
          logger: logging.Logger = None):
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: A MoleculeDataset (or a list of MoleculeDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    # debug = logger.debug if logger is not None else print

    model.train()

    # data.shuffle()

    loss_sum, iter_count = 0, 0
    cum_loss_sum, cum_iter_count = 0, 0


    mol_collator = MolCollator(shared_dict=shared_dict, args=args)

    num_workers = 4
    if type(data) == DataLoader:
        mol_loader = data
    else:
        mol_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=mol_collator)

    for _, item in enumerate(mol_loader):
        _, batch, features_batch, mask, targets = item
        if next(model.parameters()).is_cuda:
            mask, targets = mask.cuda(), targets.cuda()
        class_weights = torch.ones(targets.shape)

        if args.cuda:
            class_weights = class_weights.cuda()

        # Run model
        model.zero_grad()
        atom_vector=[]
        bond_vector=[]
        aav, abv = model.make_emb_vec(batch, features_batch)
        
        atom_vector.append(aav)
        bond_vector.append(abv)
        #if (n_iter // args.batch_size) % args.log_frequency == 0:
        #    lrs = scheduler.get_lr()
        #    loss_avg = loss_sum / iter_count
        #    loss_sum, iter_count = 0, 0
        #    lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
        
def run_training_cfm(args: Namespace, time_start, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print


    # pin GPU to local rank.
    idx = args.gpu
    if args.gpu is not None:
        torch.cuda.set_device(idx)

    features_scaler, scaler, shared_dict, test_data, train_data, val_data = load_data(args, debug, logger)
    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.multi_class:
        sum_test_preds = np.zeros((len(test_smiles), args.multi_class_num))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            if len(args.checkpoint_paths) == 1:
                cur_model = 0
            else:
                cur_model = model_idx
            debug(f'Loading model {cur_model} from {args.checkpoint_paths[cur_model]}')
            model = load_checkpoint(args.checkpoint_paths[cur_model], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = build_model(model_idx=model_idx, args=args)

        if args.fine_tune_coff != 1 and args.checkpoint_paths is not None:
            debug("Fine tune fc layer with different lr")
            initialize_weights(model_idx=model_idx, model=model.ffn, distinct_init=args.distinct_init)

        # Get loss and metric functions
        loss_func = get_loss_func(args, model)

        optimizer = build_optimizer(model, args)

        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # set up wandb
        if args.wandb : 
            wandb.init(project=args.wandb_name)
            wandb.config = args
            wandb.watch(model)


        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Bulid data_loader
        shuffle = True
        mol_collator = MolCollator(shared_dict={}, args=args)
        train_data = DataLoader(train_data,
                                batch_size=args.batch_size,
                                shuffle=shuffle,
                                num_workers=10,
                                collate_fn=mol_collator)
        
        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        min_val_loss = float('inf')
        for epoch in range(args.epochs):
            s_time = time.time()
            n_iter, train_loss = train(
                epoch=epoch,
                model=model,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                shared_dict=shared_dict,
                logger=logger
            )
            t_time = time.time() - s_time
            s_time = time.time()
            val_scores, val_loss = evaluate(
                model=model,
                data=val_data,
                loss_func=loss_func,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                shared_dict=shared_dict,
                logger=logger,
                args=args
            )
            v_time = time.time() - s_time


            # Average validation score
            avg_val_score = np.nanmean(val_scores)


            # Logged after lr step
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.6f}'.format(train_loss),
                  'loss_val: {:.6f}'.format(val_loss),
                  f'{args.metric}_val: {avg_val_score:.4f}',
                  # 'auc_val: {:.4f}'.format(avg_val_score),
                  'cur_lr: {:.5f}'.format(scheduler.get_lr()[-1]),
                  't_time: {:.4f}s'.format(t_time),
                  'v_time: {:.4f}s'.format(v_time))

            if args.tensorboard:
                writer.add_scalar('loss/train', train_loss, epoch)
                writer.add_scalar('loss/val', val_loss, epoch)
                writer.add_scalar(f'{args.metric}_val', avg_val_score, epoch)

            if args.wandb :         
                wandb.log({"val_loss" : val_loss, "val_metrics" : val_scores})


            # Save model checkpoint if improved validation score
            if args.select_by_loss:
                if val_loss < min_val_loss:
                    min_val_loss, best_epoch = val_loss, epoch
                    save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)
            else:
                if args.minimize_score and avg_val_score < best_score or \
                        not args.minimize_score and avg_val_score > best_score:
                    best_score, best_epoch = avg_val_score, epoch
                    save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

            if epoch - best_epoch > args.early_stop_epoch:
                break

        ensemble_scores = 0.0

        # Evaluate on test set using model with best validation score
        if args.select_by_loss:
            info(f'Model {model_idx} best val loss = {min_val_loss:.6f} on epoch {best_epoch}')
        else:
            info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)

        test_preds, _ = predict(
            model=model,
            data=test_data,
            loss_func=loss_func,
            batch_size=args.batch_size,
            logger=logger,
            shared_dict=shared_dict,
            scaler=scaler,
            args=args
        )

        test_scores, test_scores_AUC, test_scores_ACC, test_scores_REC, test_scores_PREC, test_scores_SPEC, test_scores_F1, test_scores_BA, test_TP, test_FP, test_TN, test_FN = evaluate_predictions_cfm(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            args=args, logger=logger
        )

        avg_test_score_AUC = np.nanmean(test_scores_AUC)
        avg_test_score_ACC = np.nanmean(test_scores_ACC)
        avg_test_score_REC = np.nanmean(test_scores_REC)
        avg_test_score_PREC = np.nanmean(test_scores_PREC)
        avg_test_score_SPEC = np.nanmean(test_scores_SPEC)
        avg_test_score_F1 = np.nanmean(test_scores_F1)
        avg_test_score_BA = np.nanmean(test_scores_BA)
        avg_test_TP = np.nanmean(test_TP)
        avg_test_TN = np.nanmean(test_TN)
        avg_test_FP = np.nanmean(test_FP)
        avg_test_FN = np.nanmean(test_FN)
        info(f'Model test AUC = {avg_test_score_AUC:.6f}')
        info(f'Model test ACC = {avg_test_score_ACC:.6f}')
        info(f'Model test REC = {avg_test_score_REC:.6f}')
        info(f'Model test PREC = {avg_test_score_PREC:.6f}')
        info(f'Model test SPEC = {avg_test_score_SPEC:.6f}')
        info(f'Model test F1 = {avg_test_score_F1:.6f}')
        info(f'Model test BA = {avg_test_score_BA:.6f}')
        info(f'Confusion matrix\nTP : {avg_test_TP:.6f}\tFP : {avg_test_FP:.6f}')
        info(f'FN : {avg_test_FN:.6f}\tTN : {avg_test_TN:.6f}')

        if args.show_individual_scores:
            # Individual test scores
            for task_name, test_score in zip(args.task_names, test_scores):
                info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')

        ind = [['preds'] * args.num_tasks + ['targets'] * args.num_tasks, args.task_names * 2]
        ind = pd.MultiIndex.from_tuples(list(zip(*ind)))
        if args.multi_class:
            data = np.concatenate([np.array([np.argmax(x) for x in test_preds]).reshape(-1,1), np.array(test_targets)], 1)
        else:
            data = np.concatenate([np.array(test_preds), np.array(test_targets)], 1)
        test_result = pd.DataFrame(data, index=test_smiles, columns=ind)
        test_result.to_csv(os.path.join(args.save_dir, 'test_result.csv'))
    
    return test_scores, test_scores_AUC, test_scores_ACC, test_scores_REC, test_scores_PREC, test_scores_SPEC, test_scores_F1, test_scores_BA, test_TP, test_FP, test_TN, test_FN