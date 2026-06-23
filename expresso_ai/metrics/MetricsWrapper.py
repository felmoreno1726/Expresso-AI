
import torch
from ignite.metrics import Accuracy, Loss, Recall, Precision, Fbeta, MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError
from ignite.contrib.metrics import ROC_AUC
from ignite.exceptions import NotComputableError

#from .clustered_roc_auc import ClusteredROCAUC
#from .clustered_f1 import ClusteredF1
#from .clustered_precision import ClusteredPrecision
#from .clustered_recal import ClusteredRecall
#from .clustered_accuracy import ClusteredAccuracy
from . import ClusteredMSE, ClusteredRMSE, ClusteredMAE
from ..utils.AverageMeter import AverageMeter

class MetricsWrapper():
    """
    Wraps a set of metrics to automate the process of computing the value of varied metrics to perform on the input
    """
    
    METRIC_NAMES = set({'clustered_mse', 'clustered_rmse', 'clustered_mae', 'mse', 'rmse', 'mae'})

    def __init__(self, metric, additional_metrics):
        """
        metric (str): name of the main metric to evaluate input/output on.
        optional metrics (arr[str]): name of metrics to also report out.
        """
        self.metric = metric
        self.additional_metrics = additional_metrics
        #assert that metrics defined are supported
        assert metric in self.METRIC_NAMES, "The metric specified is currently not supported: {}".format(metric)
        assert set(additional_metrics).issubset(self.METRIC_NAMES), "A metric specified is currently not defined {}".format(additional_metrics)
        #define metric objects
        precision_object = Precision(average=False)
        recall_object = Recall(average=False)
        F1_object = Fbeta(beta=1, precision=precision_object, recall=recall_object)
        metrics = {
                ###Classification metrics
                'roc_auc' :   ROC_AUC(),
                'F1' :      F1_object,
                'precision':precision_object,
                'recall':   recall_object,
                'accuracy': Accuracy(),
                ###TODO: define clustered classification metrics
                #'clustered_roc_auc': ClusteredROCAUC(),
                #'clustered_f1': ClusteredF1(),
                #'clustered_precision': ClusteredPrecision(),
                #'clustered_recall': ClusteredRecall(),
                #'clustered_accuracy': ClusteredAccuracy()
                ###Regression metrics
                'clustered_mse': ClusteredMSE(),
                'clustered_rmse': ClusteredRMSE(),
                'clustered_mae': ClusteredMAE(),
                'mse': MeanSquaredError(),
                'rmse': RootMeanSquaredError(),
                'mae': MeanAbsoluteError(),
        }
        #filter out mappings to only those defined in those passed to the constructor
        self.metrics = {
                metric_name: metric_function for metric_name, metric_function in metrics.items() if (metric_name == metric or metric_name in additional_metrics)
        }

    def update(self, output_target_tuple, ids=None):
        for metric_name, metric_object in self.metrics.items():
            #if the metric uses clustering
            if metric_name.find('clustered') != -1:
                #pass in ids for cluster formation
                metric_object.update((*output_target_tuple, ids))
            else:
                metric_object.update(output_target_tuple)


    def reset(self):
        """
        Resets the state of metric objects. Used at the end of each epoch
        """
        for metric_object in self.metrics.values():
            metric_object.reset()


    def compute(self):
        """
        Computes the metric score for each metric, stores them as the internal field 'computed_metrics' and returns a copy of them as a dictionary of (metric_name, score) pairs.
        Method may raise Exceptios such as NotComputableError or ValueError when a metric may not be computable.
        """
        metric_outputs = {}
        for metric_name, metric_object in self.metrics.items():
            metric_computation = metric_object.compute()
            metric_outputs[metric_name] = metric_computation.item() if torch.is_tensor(metric_computation) else metric_computation
        self.computed_metrics = metric_outputs
        return {**metric_outputs}

    def compute_metric(self):
        """
        Computes the metric score for self.metric and returns it.
        """
        return self.metrics[self.metric].compute()


    def compute_robust(self):
        """
        Computes the metric score for each metric, stores them as the internal field 'computed_metrics' and returns a copy of them as a dictionary of (metric_name, score) pairs. If a metric is not computable, excetion is catched and the score is returned as a 'nan' float.
        """
        computed_metrics = {}
        for metric_name in self.metrics.keys():
            metric_object = self.metrics[metric_name]
            try:
                metric_score = metric_object.compute()
            except NotComputableError as e:
                print(e)
                metric_score = float('nan')
            except ValueError as e:
                print(e)
                metric_score = float('nan')
            computed_metrics[metric_name] = metric_score
        self.computed_metrics = computed_metrics
        return {**computed_metrics}


    def report_metrics(self):
        report = ""
        for metric_name, metric_score in self.computed_metrics.items():
            report += "{metric_name} ({metric_score:.4f})\t".format(
                    metric_name=metric_name,
                    metric_score = metric_score)
        return report


    def get_computed_metrics(self):
        return {**self.computed_metrics}


    def report_metric(self):
        return self.report_metrics([self.metric])


    def report_additional_metrics(self):
        return self.report_metrics(self.additional_metrics)

