from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools


def make_confusion_matrix(validation_ds, model):
    """Makes a confusion matrix for the model

    Args:
        validation_ds (keras.preprocessing.image.ImageDataGenerator): validation dataset might be another type of dataset
        model (keras.engine.functional.Functional): model to be evaluated

    Returns:
        numpy.ndarray: 2D array
    """

    validation_samples = len(validation_ds)
    # Next, use the model to make predictions on the validation set
    validation_predictions = model.predict(validation_ds, steps=validation_samples)
    # Convert the predictions to class labels
    validation_predictions = np.argmax(validation_predictions, axis=-1)

    matrix = confusion_matrix(validation_ds.classes, validation_predictions)

    return matrix



def plot_confusion_matrix(matrix, class_names):
    """Makes a plot for the confusion matrix

    Args:
        matrix (numpy.ndarray): Confusion matrix
        class_names (list): A list of strings representing the class names in the order of the matrixs
    """

    plt.figure(figsize=(12, 10))

    plt.imshow(matrix, cmap='Blues')
    plt.colorbar()

    # Add labels to the plot
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    # Add a title and x and y labels
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")

    # Add grid lines
    plt.grid(False)


    # Add text to the cells
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, matrix[i, j], horizontalalignment="center", color="black")
    plt.savefig('confusion_matrix.png')
    plt.show()
