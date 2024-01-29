"""
The evaluation metrics.
"""
import math
from typing import List, Callable, Union
import numpy as np

from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, mean_absolute_error, r2_score, \
    precision_recall_curve, auc, recall_score, confusion_matrix, f1_score, precision_score, classification_report, multilabel_confusion_matrix


def accuracy(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the accuracy of a binary prediction task using a given threshold for generating hard predictions.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed accuracy.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
    return accuracy_score(targets, hard_preds)

def recall(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the recall of a binary prediction task using a given threshold for generating hard predictions.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed recall.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
    return recall_score(targets, hard_preds)

def recall_weighted(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the recall of a binary prediction task using a given threshold for generating hard predictions.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed recall.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
    return recall_score(targets, hard_preds, average='weighted')


def sensitivity(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the sensitivity of a binary prediction task using a given threshold for generating hard predictions.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed sensitivity.
    """
    return recall(targets, preds, threshold)


def specificity(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the specificity of a binary prediction task using a given threshold for generating hard predictions.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed specificity.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
    tn, fp, _, _ = confusion_matrix(targets, hard_preds).ravel()
    return tn / float(tn + fp)



def rmse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return math.sqrt(mean_squared_error(targets, preds))


def get_metric_func(metric: str, multi_class=False) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    """
    Gets the metric function corresponding to a given metric name.

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """
    # Note: If you want to add a new metric, please also update the parser argument --metric in parsing.py.
    if metric == 'auc':
        return roc_auc_score

    elif metric == 'prc-auc':
        return prc_auc

    elif metric == 'rmse':
        return rmse

    elif metric == 'mae':
        return mean_absolute_error

    elif metric == 'r2':
        return r2_score

    elif metric == 'accuracy':
        return accuracy

    elif metric == 'recall':
        return recall

    elif metric == 'recall_weighted':
        return recall_weighted

    elif metric == 'sensitivity':
        return sensitivity

    elif metric == 'specificity':
        return specificity

    elif metric == 'f1':
        return f1

    elif metric == 'f1_weighted':
        return f1_weighted


    elif metric == 'precision':
        return precision

    elif metric == 'precision_weighted':
        return precision_weighted

    raise ValueError(f'Metric "{metric}" not supported.')


def prc_auc(targets: List[int], preds: List[float]) -> float:
    """
    Computes the area under the precision-recall curve.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)

def f1(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the F1score.(HDH add this)

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed F1score.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
#    recall = recall_score(targets, hard_preds, average='macro')
#    precision = precision_score(targets, hard_preds, average='macro')
#    precision, recall, _ = precision_recall_curve(targets, preds)
#    f1 = 2*precision*recall/(precision+recall)
    return f1_score(targets, hard_preds)

def f1_weighted(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the F1score.(HDH add this)

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed F1score.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
#    recall = recall_score(targets, hard_preds, average='weighted')
#    precision = precision_score(targets, hard_preds, average='weighted')
#    f1 = 2*precision*recall/(precision+recall)
    return f1_score(targets, hard_preds, average='weighted')


def precision(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the recall of a binary prediction task using a given threshold for generating hard predictions.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed recall.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
    return precision_score(targets, hard_preds)

def precision_weighted(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the recall of a binary prediction task using a given threshold for generating hard predictions.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed recall.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
    return precision_score(targets, hard_preds, average='weighted')

def confusion_mat(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the specificity of a binary prediction task using a given threshold for generating hard predictions.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed specificity.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
    tn, fp, fn, tp = confusion_matrix(targets, hard_preds).ravel()
    acc = accuracy_score(targets, hard_preds)
    rec = recall_score(targets, hard_preds)
    prec = precision_score(targets, hard_preds)
    spe = tn / float(tn + fp)
    f1s = f1_score(targets, hard_preds)
    BA = (rec+spe)/2
    return acc, rec, prec, spe, f1s, BA, tp, fp, tn, fn

def confusion_mat_multi(targets: List[int], preds: List[float], class_num: int, threshold: float = 0.5) -> float:
    """
    Computes the specificity of a binary prediction task using a given threshold for generating hard predictions.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed specificity.
    """
    hard_preds = [np.argmax(x) for x in preds]
    result = multilabel_confusion_matrix(targets, hard_preds).ravel()
    tn=[]
    fp=[]
    fn=[]
    tp=[]
    for i in range(class_num):
        tn.append(result[4*i])
        fp.append(result[4*i+1])
        fn.append(result[4*i+2])
        tp.append(result[4*i+3])
    acc = accuracy_score(targets, hard_preds)
    rec = recall_score(targets, hard_preds, average='macro')
    prec = precision_score(targets, hard_preds, average='macro')
    f1s = f1_score(targets, hard_preds, average='macro')
    return acc, rec, prec, f1s, tp, fp, tn, fn
