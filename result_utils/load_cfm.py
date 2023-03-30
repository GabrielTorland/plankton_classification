import argparse
import tensorflow as tf
from confusion_matrix import make_confusion_matrix, plot_confusion_matrix

def main(args):
    # Load the trained model
    model = tf.keras.models.load_model(args.model)

    # Load the data
    test_ds = tf.keras.utils.image_dataset_from_directory(
    directory=args.dataset,
    labels='inferred',
    label_mode='categorical',
    shuffle=False,
    batch_size= 32,
    image_size=(224, 224)
    )

    # Create the confusion matrix
    cm = make_confusion_matrix(test_ds, model)
    
    # Plot and save the confusion matrix
    plot_confusion_matrix(cm, test_ds.class_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a confusion matrix from a trained CNN model.')
    parser.add_argument('-m', '--model', type=str, help='Path to the trained model')
    parser.add_argument('-d', '--dataset', type=str, help='Path to the test data directory')
    args = parser.parse_args()
    main(args)
