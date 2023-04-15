import argparse
import tensorflow as tf
from metrics import plot_confusion_matrix, plot_roc_curve, plot_pr_curve, print_f1_scores, top_k_error, print_precision_recall, display_auc_pr_table, print_confusion_matrix_console
import numpy as np



def main(args):
    # Load the trained model
    model = tf.keras.models.load_model(args.model)

    # Load the data
    test_ds = tf.keras.utils.image_dataset_from_directory(
    directory=args.dataset,
    labels='inferred',
    label_mode='categorical',
    shuffle=True,
    batch_size= 32,
    image_size=(224, 224)
    )


    y_true_vector = []
    y_pred_vector = []

    # Iterate through the validation dataset and collect true labels and predictions
    for x, y in test_ds:
        # y is a tensor with all the true labels. numpy() extracts the "true class vector" (e.g. [0, 0, 1])
        # Where the index of the 1 is the true class (e.g. [0, 0, 1] -> 2)
        y_true_vector.extend(y.numpy()) 
        
        # model.predict(x) returns a list of predictions vectors (e.g. [[0.1, 0.8, 0.1], [0.2, 0.1, 0.7]])
        y_pred_vector.extend(model.predict(x)) 

    # Convert the lists to numpy arrays
    y_true_vector = np.array(y_true_vector)
    y_pred_vector = np.array(y_pred_vector)

    y_true = np.argmax(y_true_vector, axis=-1)
    y_pred = np.argmax(y_pred_vector, axis=-1)


    # Plot and save the confusion matrix
    plot_confusion_matrix(y_true, y_pred, test_ds.class_names)

    # Plot and save the normalized confusion matrix
    plot_confusion_matrix(y_true, y_pred, test_ds.class_names, normalize=True)

    # Plot and save the precision-recall curve
    plot_pr_curve(y_true, y_pred_vector, test_ds.class_names)

    # Plot and save the ROC curve
    plot_roc_curve(y_true, y_pred_vector, test_ds.class_names)

    # Print the F1 scores
    print_f1_scores(y_true, y_pred, test_ds.class_names)

    # Print the top-k error
    top_1_error = top_k_error(y_true, y_pred_vector, k=1)
    top_5_error = top_k_error(y_true, y_pred_vector, k=5)
    print(f'Top-1 error: {top_1_error}')
    print(f'Top-5 error: {top_5_error}')

    # Print the precision and recall
    print_precision_recall(y_true, y_pred, test_ds.class_names)

    # Print the AUC-PR table
    table = display_auc_pr_table(y_true, y_pred_vector, test_ds.class_names)
    print(table)

    print()

    # Print the confusion matrix in the console
    print_confusion_matrix_console(y_true, y_pred, test_ds.class_names)
    print()
    
    # Print the normalized confusion matrix in the console
    print_confusion_matrix_console(y_true, y_pred, test_ds.class_names, normalize=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a confusion matrix from a trained CNN model.')
    parser.add_argument('-m', '--model', type=str, help='Path to the trained model')
    parser.add_argument('-d', '--dataset', type=str, help='Path to the test data directory')
    args = parser.parse_args()
    main(args)