# Plankton Classifier with Keras

This repository contains the code for a plankton classifier built with the Keras deep learning library. The model is trained to identify different species of plankton from images, using a dataset of 27,900 RGB FlowCam images with a resolution of 120x120x3. This work is based on a bachelor thesis by Gabriel Torland, Kim Svalde, and Andreas Primstad.

## Data

The training data consists of images of plankton and their corresponding labels. The data has been split into a training set and a validation set.

## Model

The model is a convolutional neural network (CNN) trained using the Adam optimization algorithm. The architecture of the CNN is defined in the `model.py` file.

## Training

The model is trained using the `train.py` script. The training process can be customized with various parameters such as the number of epochs and the batch size.

## Evaluation

The performance of the model is evaluated on the validation set using the `evaluate.py` script. The script provides the overall accuracy of the model as well as the precision, recall, and f1-score for each class.

## Predictions

The `predict.py` script can be used to make predictions on new images. The script provides the predicted class and the associated probability.

## Dependencies

The code in this repository requires the following packages:
- Keras
- NumPy
- SciPy

## References

This work is inspired by the following resources:
- [Deep Learning with Python by Francois Chollet](https://www.manning.com/books/deep-learning-with-python)
- [Plankton Classification with Deep Learning by Adrian Rosebrock](https://www.pyimagesearch.com/2018/12/17/plankton-
