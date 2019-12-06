import matplotlib.pyplot as plt
import numpy as np
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
    # TODO: Make plots for loss curves and accuracy curves.
    # TODO: You do not have to return the plots.
    # TODO: You can save plots as files by codes here or an interactive way according to your preference.

    plt.figure()
    plt.plot(np.arange(len(train_losses)), np.array(train_losses) * 0.53, label='Train')
    plt.plot(np.arange(len(valid_losses)), np.array(valid_losses) * 0.53, label='Validation')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(loc="best")
    plt.savefig('plot_losses')
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Train')
    plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(loc="best")
    plt.savefig('plot_accuracies')
    plt.show()


def plot_confusion_matrix_sklearn_example(y_true, y_pred, classes,
                                          normalize=False,
                                          title=None,
                                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    This function used from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plot_confusion_matrix(results, class_names):
    # TODO: Make a confusion matrix plot.
    # TODO: You do not have to return the plots.
    # TODO: You can save plots as files by codes here or an interactive way according to your preference.

    y_true = [i[0] for i in results]
    y_pred = [i[1] for i in results]
    plot_confusion_matrix_sklearn_example(y_true, y_pred, classes=class_names)
    plt.savefig('plot_cm.png')
    plt.show()


# Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def plot_roc_curve(yhat_raw, y):
    if yhat_raw.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    # get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y.shape[1]):
        # only if there are true positives for this label
        if y[:, i].sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(y[:, i], yhat_raw[:, i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = auc(fpr[i], tpr[i])
                if not np.isnan(auc_score):
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)

    n_classes = y.shape[1]

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), yhat_raw.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    lw = 2
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #                    ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC multi-class')
    plt.legend(loc="lower right")
    plt.savefig('plot_roc')
    plt.show()
