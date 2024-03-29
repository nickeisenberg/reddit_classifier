from matplotlib.figure import Figure
import numpy as np
from matplotlib import colormaps
import matplotlib.pyplot as plt


class ConfusionMatrix:
    def __init__(self, labels):
        self.labels = labels


    def run(self, targets, predictions, save_to):
        matrix = self.compute_confusion_matrix(targets, predictions)
        fig = self.make_confusion_matrix_fig(matrix, self.labels)
        fig.savefig(save_to)
        return None

    
    @staticmethod
    def compute_confusion_matrix(targets, predictions):
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
    def make_confusion_matrix_fig(matrix, class_names) -> Figure:
        """
        Plot the confusion matrix using matplotlib.
    
        Parameters:
        - matrix: A 2D numpy array representing the confusion matrix.
        - class_names: A list of names for the classes.
        """
        try:
            class_names = list(np.array(class_names).astype(str))
        except:
            raise Exception("class names could not be converted to string")
    
        accuracy = np.round(np.einsum("ii->i", matrix).sum() / matrix.sum() * 100, 2)
    
        fig, ax = plt.subplots(figsize=(10, 7))
        percentages = matrix / matrix.sum(axis=1)
        cax = ax.matshow(percentages, cmap=colormaps["Blues"])
        plt.title(f'Confusion Matrix: Accuracy {accuracy}%', pad=20)
        fig.colorbar(cax)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names)
    
        # Loop over data dimensions and create text annotations.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j, i, 
                    f"{np.round(percentages[i, j] * 100 ,2 )}%", 
                    ha="center", va="center", color="Black",
                    size=12
                )
    
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
    
        return fig


def confusion_matrix(targets, predictions, labels, save_to):
    matrix = compute_confusion_matrix(targets, predictions)
    fig = make_confusion_matrix_fig(matrix, labels)
    fig.savefig(save_to)
    return None


def compute_confusion_matrix(targets, predictions):
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


def make_confusion_matrix_fig(matrix, class_names) -> Figure:
    """
    Plot the confusion matrix using matplotlib.

    Parameters:
    - matrix: A 2D numpy array representing the confusion matrix.
    - class_names: A list of names for the classes.
    """
    try:
        class_names = list(np.array(class_names).astype(str))
    except:
        raise Exception("class names could not be converted to string")

    accuracy = np.round(np.einsum("ii->i", matrix).sum() / matrix.sum() * 100, 2)

    fig, ax = plt.subplots(figsize=(10, 7))
    percentages = matrix / matrix.sum(axis=1)
    cax = ax.matshow(percentages, cmap=colormaps["Blues"])
    plt.title(f'Confusion Matrix: Accuracy {accuracy}%', pad=20)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)

    # Loop over data dimensions and create text annotations.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, 
                f"{np.round(percentages[i, j] * 100 ,2 )}%", 
                ha="center", va="center", color="Black",
                size=12
            )

    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    return fig


if __name__ == "__main__":
    conf_mat = ConfusionMatrix(np.arange(3))
    conf_mat.run(
        np.random.randint(0, 2, 10),
        np.random.randint(0, 2, 10),
        "confmat.png"
    )



