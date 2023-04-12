from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.optimizers import Adam


DEFAULT_INPUT_SHAPE = (224, 224, 3)
DEFAULT_WEIGHTS = "imagenet"


def freeze_base_model(model):
    """ Freeze the base model.

    Args:
        model (tensorflow.keras.applications.*): Base model. 
    """    
    for layer in model.layers:
        layer.trainable = False

        
        
        
def create_mobilenetv3_large_model_no_fc(num_classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """
    Create a new MobilenetV3Large model with transfer learning. 

    Args:
        weights (str, optional): Initial weights. Defaults to "imagenet".
        input_shape (tuple, optional): Shape of input images. Defaults to (224, 224, 3).

    Returns:
        tf.keras.application.MobileNetV3Large : Model. 
    """

    # create a base model
    feature_extractor = MobileNetV3Large(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        input_shape=input_shape # specify input shape
    ) 

    # add fully connected layers after the base model (i.e., at the end of the network)
    x = feature_extractor.output 

    # change the spatial dimensions to 1x1
    # thus, the output becomes (batch_size, 1, 1, 2048)
    x = GlobalAveragePooling2D()(x)

    # final output layer
    predictions = Dense(num_classes, activation='softmax')(x)

    # create the model
    model = Model(inputs=feature_extractor.input, outputs=predictions)
    
    # compile the model
    model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

    return model 


def create_mobilenetv3_small_model_no_fc(num_classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """
    Create a new MobileNetV3Small model with transfer learning. 

    Args:
        weights (str, optional): Initial weights. Defaults to "imagenet".
        input_shape (tuple, optional): Shape of input images. Defaults to (224, 224, 3).

    Returns:
        tf.keras.application.MobileNetV3Small : Model. 
    """

    # create a base model
    feature_extractor = MobileNetV3Small(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        input_shape=input_shape # specify input shape
    ) 

    # add fully connected layers after the base model (i.e., at the end of the network)
    x = feature_extractor.output 

    # change the spatial dimensions to 1x1
    # thus, the output becomes (batch_size, 1, 1, 2048)
    x = GlobalAveragePooling2D()(x)

    # final output layer
    predictions = Dense(num_classes, activation='softmax')(x)

    # create the model
    model = Model(inputs=feature_extractor.input, outputs=predictions)
    
    # compile the model
    model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

    return model 
        
        
        
        

def create_mobilenetv3_large_model(num_classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """
    Create a new MobilenetV3Large model with transfer learning. 

    Args:
        weights (str, optional): Initial weights. Defaults to "imagenet".
        input_shape (tuple, optional): Shape of input images. Defaults to (224, 224, 3).

    Returns:
        tf.keras.application.MobileNetV3Large : Model. 
    """

    # create a base model
    feature_extractor = MobileNetV3Large(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        input_shape=input_shape # specify input shape
    ) 

    # freeze the base model
    freeze_base_model(feature_extractor)

    # add fully connected layers after the base model (i.e., at the end of the network)
    x = feature_extractor.output 

    # change the spatial dimensions to 1x1
    # thus, the output becomes (batch_size, 1, 1, 2048)
    x = GlobalAveragePooling2D()(x)

    # fully connected layer for intermediate features
    x = Dense(1024, activation='relu')(x)

    # dropout layer to prevent overfitting
    x = Dropout(0.5)(x)

    # final output layer
    predictions = Dense(num_classes, activation='softmax')(x)

    # create the model
    model = Model(inputs=feature_extractor.input, outputs=predictions)
    
    # compile the model
    model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

    return model 


def create_mobilenetv3_small_model(num_classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """
    Create a new MobileNetV3Small model with transfer learning. 

    Args:
        weights (str, optional): Initial weights. Defaults to "imagenet".
        input_shape (tuple, optional): Shape of input images. Defaults to (224, 224, 3).

    Returns:
        tf.keras.application.MobileNetV3Small : Model. 
    """

    # create a base model
    feature_extractor = MobileNetV3Small(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        input_shape=input_shape # specify input shape
    ) 

    # freeze the base model
    freeze_base_model(feature_extractor)

    # add fully connected layers after the base model (i.e., at the end of the network)
    x = feature_extractor.output 

    # change the spatial dimensions to 1x1
    # thus, the output becomes (batch_size, 1, 1, 2048)
    x = GlobalAveragePooling2D()(x)

    # fully connected layer for intermediate features
    x = Dense(1024, activation='relu')(x)

    # dropout layer to prevent overfitting
    x = Dropout(0.5)(x)

    # final output layer
    predictions = Dense(num_classes, activation='softmax')(x)

    # create the model
    model = Model(inputs=feature_extractor.input, outputs=predictions)
    
    # compile the model
    model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

    return model 