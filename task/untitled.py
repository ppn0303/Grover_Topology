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
            makedirs(args.save_dir)
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

