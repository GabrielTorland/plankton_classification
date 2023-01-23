from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools


def make_confusion_matrix(validation_ds, model):
    """
    validation_ds: A keras.preprocessing.image.ImageDataGenerator object
    model: A keras.engine.functional.Functional object

    model and validation_ds might be other types, but this is what I used.

    returns: A numpy.ndarray object
    
    """

    validation_samples = len(validation_ds)
    # Next, use the model to make predictions on the validation set
    validation_predictions = model.predict(validation_ds, steps=validation_samples)
    # Convert the predictions to class labels
    validation_predictions = np.argmax(validation_predictions, axis=-1)

    matrix = confusion_matrix(validation_ds.classes, validation_predictions)

    return matrix



def plot_confusion_matrix(matrix, class_names):
    """
    matrix: A numpy.ndarray object
    class_names: A list of strings representing the class names in the order of the matrix
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
