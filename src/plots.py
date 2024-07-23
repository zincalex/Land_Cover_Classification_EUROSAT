import matplotlib.pyplot as plt     # plots
import numpy as np                  # arrays 
from sklearn import metrics


def show_confusion_matrix(true_labels, predictions, title = "Confusion Matrix") :
    """Print and show the confusion matrix  
 
    Args:
        true_labels (list or numpy array):      true labels
        predictions (list or numpy array):      predicted labels
        title (string):                         title displayed on screen
    """ 
    disp = metrics.ConfusionMatrixDisplay.from_predictions(true_labels, predictions)
    disp.figure_.suptitle(title)
    print(f"Confusion matrix: \n{disp.confusion_matrix}")
    plt.tight_layout()
    plt.show()


def plot_train_loss(losses, epochs, sub_band_names = None) :
    """Show the evolution of the training loss with respect to the epochs. 
 
    Args:
        losses (list):              can be a list of train loss values, or a list of lists with train loss values
        epochs (int):               number of epochs done during training
        sub_band_names (list):      list of strings
    """ 
    plt.figure(figsize=(10, 6))
    x_range = range(1, epochs + 1)
    colors = ["blue", "green", "red", "purple"]     # Only four colors since we have max 4 classifiers
    
    if len(losses) == epochs : # Case when only 1 resnet is trained
        plt.plot(x_range, losses, label='ResNet Classifier', color=colors[0])
        plt.title('PCA Analysis')
    else:
        for i, loss in enumerate(losses) : # Case when we pass the training losses of the ensemble of classifiers
            plt.plot(x_range, loss, label=f'ResNet Classifier on {sub_band_names[i]}', color=colors[i])
        plt.title('Ensemble Resnet50s Analysis')

    plt.xticks(np.arange(0, epochs + 1, step=2))   
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.legend(loc="upper right")
    plt.show()

