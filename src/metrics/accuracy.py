from torch import Tensor, tensor, hstack
from numpy import round


class Accuracy:
    def __init__(self):
        self.accuracy = 0.0
        self._running_total = tensor([])

    def log(self, predictions: Tensor, targets: Tensor):
        predictions, targets = predictions.cpu(), targets.cpu()
        acc = ((predictions == targets) * 1).float()
        self._running_total = hstack([self._running_total, acc])
        self.accuracy = round(
             self._running_total.mean().item() * 100, 2
         )

        
    def reset_state(self, *args, **kwargs):
        """
        do some things here and then reset the state of the metric.
        """
        self._running_total = tensor([])

