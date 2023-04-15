from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, precision_score, recall_score, f1_score, average_precision_score
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd




def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False):
    """Makes a plot for the confusion matrix

    Args:
        y_true (list): A list of integers representing the true classes
        y_pred (list): A list of integers representing the predicted classes
        class_names (list): A list of strings representing the class names in the order of the matrix
        normalize (bool): Whether to normalize the confusion matrix. Defaults to False.
    """

    matrix = confusion_matrix(y_true, y_pred)
    
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

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
    plt.title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")

    # Add grid lines
    plt.grid(False)

    # Add text to the cells
    fmt = '.2f' if normalize else 'd'
    threshold = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        cell_value = matrix[i, j]
        plt.text(j, i, format(cell_value, fmt),
                 horizontalalignment="center",
                 color="white" if cell_value > threshold else "black")
    
    plt.savefig('confusion_matrix.png' if not normalize else 'normalized_confusion_matrix.png', bbox_inches='tight')
    plt.show()



def print_confusion_matrix_console(y_true, y_pred, class_names, normalize=False):
    """Makes a plot for the confusion matrix

    Args:
        y_true (list): A list of integers representing the true classes
        y_pred (list): A list of integers representing the predicted classes
        class_names (list): A list of strings representing the class names in the order of the matrix
        normalize (bool): Whether to normalize the confusion matrix. Defaults to False.
    """
        
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = "{:.2f}"
    else:
        fmt = "{:d}"

    # Print the confusion matrix with class names
    header = " " * (len(max(class_names, key=len)) + 2)
    for name in class_names:
        header += f"{name:>{len(fmt)}} "
    print(header)

    for i, (name, row) in enumerate(zip(class_names, cm)):
        row_text = f"{name:>{len(max(class_names, key=len)) + 1}} "
        row_text += " ".join([fmt.format(value) for value in row])
        print(row_text)


def plot_pr_curve(y_true, y_pred, class_names):
    """Makes a plot for the precision-recall curve

    Args:
        y_true (ndarray): A numpy array of integers representing the true classes
        y_pred (ndarray): A numpy array of floats representing the predicted probabilities
        class_names (list): A list of strings representing the class names
    """


    # Compute the precision-recall curve for each class
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        # Get the true labels and predicted scores for the current class
        y_true_class = [1 if label == i else 0 for label in y_true]
        y_pred_class = y_pred[:, i]

        # Compute the precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true_class, y_pred_class)


        # Plot the precision-recall curve for the current class
        plt.plot(recall, precision, label=class_names[i])

    # Add labels and legend to the plot
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('pr-curve.png', bbox_inches='tight')
    plt.show()




def plot_roc_curve(y_true, y_pred, class_names):
    """Makes a plot for the receiver operating characteristic (ROC) curve

    Args:
        y_true (ndarray): A numpy array of integers representing the true classes
        y_pred (ndarray): A numpy array of floats representing the predicted probabilities
        class_names (list): A list of strings representing the class names
    """

    plt.figure(figsize=(10, 8))
    
    for i in range(len(class_names)):
        y_true_class = [1 if label == i else 0 for label in y_true]
        y_pred_class = y_pred[:, i]
        
        fpr, tpr, _ = roc_curve(y_true_class, y_pred_class)
        
        plt.plot(fpr, tpr, label=f'{class_names[i]}')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('roc-curve.png', bbox_inches='tight')
    plt.show()



def print_precision_recall(y_true, y_pred, class_names):
    """Prints the precision and recall for each class and the macro and micro averages

    Args:
        y_true (ndarray): A numpy array of integers representing the true classes
        y_pred (ndarray): A numpy array of floats representing the predicted probabilities
        class_names (list): A list of strings representing the class names
    """

    n_classes = len(class_names)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    print("Precision and Recall per class:")
    for i in range(n_classes):
        print(f"{class_names[i]}: Precision = {precision[i]:.2f}, Recall = {recall[i]:.2f}")

    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"\nPrecision (macro): {precision_macro:.2f}")
    print(f"Recall (macro): {recall_macro:.2f}")

    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)

    print(f"\nPrecision (micro): {precision_micro:.2f}")
    print(f"Recall (micro): {recall_micro:.2f}")




def top_k_error(y_true, y_pred_probs, k=1):
    """Computes the top-k error

    Args:
        y_true (ndarray): A numpy array of integers representing the true classes
        y_pred (ndarray): A numpy array of floats representing the predicted probabilities
        k (int): The number of top predictions to consider
    """

    top_k_preds = np.argsort(y_pred_probs, axis=-1)[:, -k:]
    match_array = np.any(top_k_preds == y_true[:, None], axis=-1)
    error = 1 - np.mean(match_array)
    return error



def print_f1_scores(y_true, y_pred, class_names):
    """Prints the F1 score for each class and the macro and micro averages

    Args:
        y_true (ndarray): A numpy array of integers representing the true classes
        y_pred (ndarray): A numpy array of floats representing the predicted probabilities
        class_names (list): A list of strings representing the class names
    """
    n_classes = len(class_names)
    f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)

    print("F1 Score per class:")
    for i in range(n_classes):
        print(f"{class_names[i]}: F1 Score = {f1_scores[i]:.2f}")

    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"\nF1 Score (macro): {f1_macro:.2f}")

    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    print(f"\nF1 Score (micro): {f1_micro:.2f}")



def calculate_auc_pr(y_true, y_pred_probs, class_names):
    n_classes = len(class_names)
    auc_pr_scores = []

    for i in range(n_classes):
        y_true_class = [1 if label == i else 0 for label in y_true]
        y_pred_class = y_pred_probs[:, i]
        auc_pr_score = average_precision_score(y_true_class, y_pred_class)
        auc_pr_scores.append(auc_pr_score)

    y_true_one_hot = np.eye(n_classes)[y_true]
    auc_pr_micro = average_precision_score(y_true_one_hot, y_pred_probs, average='micro')
    auc_pr_macro = average_precision_score(y_true_one_hot, y_pred_probs, average='macro')

    return auc_pr_scores, auc_pr_micro, auc_pr_macro


def display_auc_pr_table(y_true, y_pred_probs, class_names):
    auc_pr_scores, auc_pr_micro, auc_pre_macro = calculate_auc_pr(y_true, y_pred_probs, class_names)

    data = {'Class': class_names, 'AUC-PR': auc_pr_scores}
    data['Class'].append('Micro-average')
    data['AUC-PR'].append(auc_pr_micro)
    data['Class'].append('Macro-average')
    data['AUC-PR'].append(auc_pre_macro)


    df = pd.DataFrame(data)
    return df