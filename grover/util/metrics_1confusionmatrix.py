"""
The evaluation metrics.
"""
import math
from typing import List, Callable, Union

from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, mean_absolute_error, r2_score, \
    precision_recall_curve, auc, recall_score, confusion_matrix, f1_score, precision_score, classification_report


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


def get_metric_func(metric: str) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    """
    Gets the metric function corresponding to a given metric name.

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """
    # Note: If you want to add a new metric, please also update the parser argument --metric in parsing.py.
    if metric == 'auc':
        return roc_auc_score

    if metric == 'prc-auc':
        return prc_auc

    if metric == 'rmse':
        return rmse

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score

    if metric == 'accuracy':
        return accuracy

    if metric == 'recall':
        return recall

    if metric == 'recall_weighted':
        return recall_weighted

    if metric == 'sensitivity':
        return sensitivity

    if metric == 'specificity':
        return specificity

    if metric == 'f1':
        return f1

    if metric == 'f1_weighted':
        return f1_weighted


    if metric == 'precision':
        return precision

    if metric == 'precision_weighted':
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
    acc = (tp+tn)/(tn+fp+fn+tp)
    rec = tp/(tp+fn)
    prec = tp/(tp+fp)
    f1s = 2*rec*prec/(rec+prec)
    results = [tn, fp, fn, tp]
    return acc, rec, prec, f1s, tp, fp, tn, fn
