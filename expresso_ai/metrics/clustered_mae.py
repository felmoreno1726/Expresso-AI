#import pandas as pd
import torch

from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

__all__ = ["ClusteredMAE"]

class ClusteredMAE(Metric):

    def __init__(self, output_transform=lambda x: x):
        self.clusters = None
        self._sum_of_absolute_errors = None
        self._num_examples = None
        super(ClusteredMAE, self).__init__(output_transform=output_transform)

    def reset(self):
        self._sum_of_absolute_errors = 0
        self._num_examples = 0
        self.clusters = {}

    def update(self, output):
        y_pred, y_label, _id = output
        for y_pred_sample, y_label_sample, _id_sample in zip(y_pred, y_label, _id):
            data_entry = self.clusters.get(_id_sample.item(), self.create_empty_entry(_id_sample.item(), y_label_sample))
            #assert that the participants stored label matches that passed in
            #assert y_label_sample == data_entry["label"], "Labels don't match. Expected label for participant {} to be {} but got {}".format(_id, data_entry["label"], y_label_sample)
            #update data_entry of a particular _id
            #each id is given the score of the average of all predictions given to it
            data_entry["score"] = (data_entry["score"] * data_entry["samples"] + y_pred_sample)/(data_entry["samples"] + 1)
            data_entry["samples"] += 1
            old_absolute_error = data_entry["absolute_error"]
            new_absolute_error = torch.abs(data_entry["label"] - data_entry["score"])
            #update running sum
            self._sum_of_absolute_errors += (new_absolute_error - old_absolute_error).item()
            #update absolute error
            data_entry["absolute_error"] = new_absolute_error
            #update cluster with current data entry
            self.clusters[_id_sample.item()] = data_entry
        #update computation
        self.num_samples = len(self.clusters)

    def compute(self):
        if self._num_samples ==0:
            raise NotComputableError
        return self._sum_of_absolute_errors / self.num_samples
    
    @staticmethod
    def create_empty_entry(_id, label):
        return {
                "label": label,
                "samples": 0,
                "score": 0,
                "absolute_error": 0
                }

