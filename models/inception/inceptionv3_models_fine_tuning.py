from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.applications import inception_v3
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

def create_inceptionv3_model(num_classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """
    Create a new InceptionV3 model with transfer learning.
    This model is based on the 2014 paper "Rethinking the Inception Architecture for Computer Vision" by Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna.
    Available at: https://arxiv.org/abs/1512.00567

    Args:
        num_classes (int): Number of classes
        weights (str, optional): Initial weights. Defaults to "imagenet".
        input_shape (tuple, optional): Shape of input images. Defaults to (224, 224).
    
        
    Returns:
        tf.keras.application.InceptionV3 : Model. 
    """

    # create a base model
    feature_extractor = inception_v3.InceptionV3(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        pooling=None, # pooling is set to None
        input_shape=input_shape # specify input shape
    )

    # freeze the base model
    freeze_base_model(feature_extractor)

    x = feature_extractor.output

    # Global Average Pooling
    x = GlobalAveragePooling2D(name='avg_pool')(x)

    # fully connected layer 1
    x = Dense(1024, activation='relu')(x)

    # final output layer
    predictions = Dense(num_classes, activation='softmax')(x)

    # create the model
    model = Model(inputs=feature_extractor.input, outputs=predictions)

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    return model