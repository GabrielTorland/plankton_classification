from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

DEFAULT_INPUT_SHAPE = (224, 224, 3)
DEFAULT_WEIGHTS = "imagenet"

def freeze_base_model(model):
    """ Freeze the base model.

    Args:
        model (tensorflow.keras.applications.*): Base model. 
    """    
    for layer in model.layers:
        layer.trainable = True

def create_vgg16_model(num_classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """
    Create a new VGG16 model with transfer learning.
    This model is based on the 2014 paper "Very Deep Convolutional Networks for Large-Scale Image Recognition" by Karen Simonyan and Andrew Zisserman.
    Available at: https://arxiv.org/abs/1409.1556

    Args:
        weights (str, optional): Initial weights. Defaults to "imagenet.
        input_shape (tuple, optional): Shape of input images. Defaults to (224, 224, 3).

    Returns:
        tf.keras.application.VGG16 : Model. 
    """

    # create a base model
    feature_extractor = VGG16(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        pooling=None, # use max pooling
        input_shape=input_shape # specify input shape
    )

    # freeze the base model
    freeze_base_model(feature_extractor)

    x = feature_extractor.output

    # Flatten the output to 1D
    x = Flatten(name='flatten')(x)

    # fully connected layer 1
    x = Dense(64, activation='relu')(x)

    # final output layer
    predictions = Dense(num_classes, activation='softmax')(x)

    # create the model
    model = Model(inputs=feature_extractor.input, outputs=predictions)

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    return model

