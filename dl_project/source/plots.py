import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

	ax1.plot(train_losses,'C0',label='Training Loss')
	ax1.plot(valid_losses,'C1',label='Validation Loss')
	ax1.set(xlabel='epoch', ylabel='Loss')
	ax1.legend(loc='upper right')
	ax1.set(title="Loss Curve")

	ax2.plot(train_accuracies, 'C0', label='Training Accuracy')
	ax2.plot(valid_accuracies, 'C1', label='Validation Accuracy')
	ax2.set(xlabel='epoch', ylabel='Accuracy')
	ax2.legend(loc='upper left')
	ax2.set(title="Accuracy Curve")

	plt.savefig('plot_learning_curves.png')

	plt.show()
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.

# Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	y_true = [i[0] for i in results]
	y_pred = [i[1] for i in results]
	cm = confusion_matrix(y_true, y_pred)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalized confusion matrix
	# print(cm)
	cmap = plt.cm.Blues
	title = "Normalized Confusion Matrix"
	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   # ... and label them with the respective list entries
		   xticklabels=class_names, yticklabels=class_names,
		   title=title,
		   ylabel='True',
		   xlabel='Predicted')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	plt.savefig('plot_confusion_matrix.png')
	plt.show()
