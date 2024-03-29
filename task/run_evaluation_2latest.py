"""
The evaluation function.
"""
from argparse import Namespace
from logging import Logger
from typing import List

import numpy as np
import torch
import torch.utils.data.distributed

from grover.data.scaler import StandardScaler
from grover.util.utils import get_class_sizes, get_data, split_data, get_task_names, get_loss_func
from grover.util.utils import load_checkpoint
from task.predict import evaluate_predictions, evaluate_predictions_cfm
from grover.util.metrics import get_metric_func
from grover.util.nn_utils import param_count
from task.predict import predict


def run_evaluation(args: Namespace, logger: Logger = None) -> List[float]:
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

    torch.cuda.set_device(0)

    # Get data
    debug('Loading data')
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')

    # Split data
    debug(f'Splitting data with seed {args.seed}')

    train_data, val_data, test_data = split_data(data=data,
                                                 split_type=args.split_type,
                                                 sizes=args.split_sizes,
                                                 seed=args.seed,
                                                 args=args,
                                                 logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)

    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler  (regression only)
    scaler = None
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        _, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)

        val_targets = val_data.targets()
        scaled_val_targets = scaler.transform(val_targets).tolist()
        val_data.set_targets(scaled_val_targets)

    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Load/build model
    if args.checkpoint_paths is not None:
        cur_model = args.seed
        target_path = []
        for path in args.checkpoint_paths:
            if "fold_%d" % cur_model in path:
                target_path = path
        debug(f'Loading model {args.seed} from {target_path}')
        model = load_checkpoint(target_path, current_args=args, cuda=args.cuda, logger=logger)
        # Get loss and metric functions
        loss_func = get_loss_func(args, model)

    debug(f'Number of parameters = {param_count(model):,}')

    test_preds, _ = predict(
        model=model,
        data=test_data,
        batch_size=args.batch_size,
        loss_func=loss_func,
        logger=logger,
        shared_dict={},
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
    info(f'Model test {args.metric} = {avg_test_score:.6f}')

    if args.show_individual_scores:
        # Individual test scores
        for task_name, test_score in zip(args.task_names, test_scores):
            info(f'Model test {task_name} {args.metric} = {test_score:.6f}')

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

    # If you want to save the prediction result, uncomment these lines.
    # ind = [['preds'] * args.num_tasks + ['targets'] * args.num_tasks, args.task_names * 2]
    # ind = pd.MultiIndex.from_tuples(list(zip(*ind)))
    # data = np.concatenate([np.array(avg_test_preds), np.array(test_targets)], 1)
    # test_result = pd.DataFrame(data, index=test_smiles, columns=ind)
    # test_result.to_csv(os.path.join(args.save_dir, 'test_result.csv'))

    return ensemble_scores


def run_evaluation_cfm(args: Namespace, logger: Logger = None) -> List[float]:
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

    torch.cuda.set_device(0)

    # Get data
    debug('Loading data')
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')

    # Split data
    debug(f'Splitting data with seed {args.seed}')

    train_data, val_data, test_data = split_data(data=data,
                                                 split_type=args.split_type,
                                                 sizes=[0.8, 0.1, 0.1],
                                                 seed=args.seed,
                                                 args=args,
                                                 logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data, args)
        debug('Class sizes')
        if args.multi_class : 
            for i, task_class_sizes in enumerate(class_sizes):
                debug(f'{i}th class size '
                      f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

        else : 
            for i, task_class_sizes in enumerate(class_sizes):
                debug(f'{args.task_names[i]} '
                      f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)

    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler  (regression only)
    scaler = None
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        _, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)

        val_targets = val_data.targets()
        scaled_val_targets = scaler.transform(val_targets).tolist()
        val_data.set_targets(scaled_val_targets)

    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.multi_class and args.dataset_type=='classification':
        sum_test_preds = np.zeros((len(test_smiles), args.multi_class_num))
    else : 
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Load/build model
    if args.checkpoint_paths is not None:
        cur_model = args.seed
        target_path = []
        for path in args.checkpoint_paths:
            if "fold_%d" % cur_model in path:
                target_path = path
        debug(f'Loading model {args.seed} from {target_path}')
        model = load_checkpoint(target_path, current_args=args, cuda=args.cuda, logger=logger)
        # Get loss and metric functions
        loss_func = get_loss_func(args, model)

    debug(f'Number of parameters = {param_count(model):,}')

    test_preds, _ = predict(
        model=model,
        data=test_data,
        batch_size=args.batch_size,
        loss_func=loss_func,
        logger=logger,
        shared_dict={},
        scaler=scaler,
        args=args
    )

    _, test_scores_AUC, test_scores_ACC, test_scores_REC, test_scores_PREC, test_scores_SPEC, test_scores_F1, test_scores_BA, test_TP, test_FP, test_TN, test_FN = evaluate_predictions_cfm(
        preds=test_preds,
        targets=test_targets,
        num_tasks=args.num_tasks,
        metric_func=metric_func,
        dataset_type=args.dataset_type,
        args=args, logger=logger
    )

    if len(test_preds) != 0:
        sum_test_preds += np.array(test_preds, dtype=float)

    # Average test score
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
            info(f'Model test {task_name} {args.metric} = {test_score:.6f}')

    # If you want to save the prediction result, uncomment these lines.
    # ind = [['preds'] * args.num_tasks + ['targets'] * args.num_tasks, args.task_names * 2]
    # ind = pd.MultiIndex.from_tuples(list(zip(*ind)))
    # data = np.concatenate([np.array(avg_test_preds), np.array(test_targets)], 1)
    # test_result = pd.DataFrame(data, index=test_smiles, columns=ind)
    # test_result.to_csv(os.path.join(args.save_dir, 'test_result.csv'))
    
    return test_scores_AUC, test_scores_ACC, test_scores_REC, test_scores_PREC, test_scores_SPEC, test_scores_F1, test_scores_BA, test_TP, test_FP, test_TN, test_FN
