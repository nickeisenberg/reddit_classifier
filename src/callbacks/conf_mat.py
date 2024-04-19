import os

import numpy as np
from numpy import ndarray

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import colormaps

from torch import Tensor

from ..trainer.trainer import Trainer
from .base import Callback


class ConfusionMatrix(Callback):
    def __init__(self, 
                 labels: list, 
                 save_root: str, 
                 figsize: tuple[int, int] = (12, 8)):
        self.labels = labels

        self.save_root = save_root

        self.figsize = figsize
        
        self.predictions = []
        self.targets = []
     

    def log(self, predictions: Tensor, targets: Tensor):
        self.predictions += predictions.tolist()
        self.targets += targets.tolist()


    def before_all_epochs(self, trainer: Trainer, *args, **kwargs):
        assert hasattr(trainer, "train_module")
        assert hasattr(trainer, "which_pass")
        assert hasattr(trainer, "current_epoch")


    def after_train_epoch_pass(self, trainer: Trainer, *args, **kwargs):
        self.reset_state(trainer.which_pass, trainer.current_epoch)


    def after_validation_epoch_pass(self, trainer: Trainer, *args, **kwargs):
        self.reset_state(trainer.which_pass, trainer.current_epoch)


    def after_evaluation_epoch_pass(self, trainer: Trainer, *args, **kwargs):
        self.reset_state(trainer.which_pass, 0)


    def reset_state(self, which, epoch, *args, **kwargs):
        matrix = self.compute_confusion_matrix(self.targets, self.predictions)
        fig = self.make_confusion_matrix_fig(matrix, self.labels, self.figsize)

        save_to = os.path.join(
            self.save_root, f"{which}_ep{epoch}.png"
        )
        fig.savefig(save_to)

        self.predictions = []
        self.targets = []
        return None

    
    @staticmethod
    def compute_confusion_matrix(targets: list[int], predictions: list[int]):
        """
        Compute a confusion matrix for multi-class classification.
    
        Parameters:
        - targets: A list of actual target values.
        - predictions: A list of predictions.
        - num_classes: The number of classes in the classification problem.
    
        Returns:
        - A 2D numpy array representing the confusion matrix.
        """
        num_classes = len(np.unique(targets))
        matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        for actual, predicted in zip(targets, predictions):
            matrix[actual, predicted] += 1
        return matrix


    @staticmethod
    def make_confusion_matrix_fig(matrix: ndarray,
                                  class_names: list,
                                  figsize: tuple[int, int] = (12, 8)) -> Figure:
        """
        Plot the confusion matrix using matplotlib.
    
        Parameters:
        - matrix: A 2D numpy array representing the confusion matrix.
        - class_names: A list of names for the classes.
        - figsize: Figure dimension tuple (width, height).
        """
        class_names = [str(name) for name in class_names]
        accuracy = np.round(np.trace(matrix) / np.sum(matrix) * 100, 2)
    
        fig, ax = plt.subplots(figsize=figsize)
        row_sums = matrix.sum(axis=1)
        percentages = np.divide(matrix, row_sums[:, None], where=row_sums[:, None] != 0)
        cax = ax.matshow(percentages, cmap='Blues')
        plt.title(f'Confusion Matrix: Accuracy {accuracy}%', pad=20)
        fig.colorbar(cax)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
    
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if row_sums[i] != 0:
                    percentage_text = f"{np.round(percentages[i, j] * 100, 2)}%" 
                else: 
                    percentage_text = '0%'

                ax.text(
                    j, i, percentage_text, 
                    ha="center", va="center", 
                    color="white" if percentages[i, j] > 0.5 else "black"
                )
    
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()

        return fig


    # @staticmethod 
    # def make_confusion_matrix_fig(matrix:ndarray, 
    #                               class_names: list,
    #                               figsize: tuple[int, int] = (12, 8)) -> Figure:
    #     """
    #     Plot the confusion matrix using matplotlib.
    # 
    #     Parameters:
    #     - matrix: A 2D numpy array representing the confusion matrix.
    #     - class_names: A list of names for the classes.
    #     """
    #     try:
    #         class_names = list(np.array(class_names).astype(str))
    #     except:
    #         raise Exception("class names could not be converted to string")
    # 
    #     accuracy = np.round(np.einsum("ii->i", matrix).sum() / matrix.sum() * 100, 2)
    # 
    #     fig, ax = plt.subplots(figsize=figsize)
    #     percentages = matrix / matrix.sum(axis=1)
    #     cax = ax.matshow(percentages, cmap=colormaps["Blues"])
    #     plt.title(f'Confusion Matrix: Accuracy {accuracy}%', pad=20)
    #     fig.colorbar(cax)
    #     ax.set_xticks(np.arange(len(class_names)))
    #     ax.set_yticks(np.arange(len(class_names)))
    #     ax.set_xticklabels(class_names, rotation=45)
    #     ax.set_yticklabels(class_names)
    # 
    #     # Loop over data dimensions and create text annotations.
    #     for i in range(matrix.shape[0]):
    #         for j in range(matrix.shape[1]):
    #             ax.text(
    #                 j, i, 
    #                 f"{np.round(percentages[i, j] * 100 ,2 )}%", 
    #                 ha="center", va="center", color="Black",
    #                 size=12
    #             )
    # 
    #     plt.xlabel('Predicted')
    #     plt.ylabel('Actual')
    #     plt.tight_layout()
    # 
    #     return fig


