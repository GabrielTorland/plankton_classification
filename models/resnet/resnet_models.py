from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet152, ResNet152V2
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


def create_resnet152_model(num_classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """
    Create a new ResNet152 model with transfer learning. 
    This model is based on the 2015 paper "Deep Residual Learning for Image Recognition" by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
    Avilable at: https://arxiv.org/abs/1512.03385

    Args:
        weights (str, optional): Initial weights. Defaults to "imagenet".
        input_shape (tuple, optional): Shape of input images. Defaults to (224, 224, 3).

    Returns:
        tf.keras.application.ResNet152 : Model. 
    """

    # create a base model
    feature_extractor = ResNet152(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        input_shape=input_shape # specify input shape
    ) 

    # freeze the base model
    freeze_base_model(feature_extractor)

    x = feature_extractor.output 

    # change the spatial dimensions to 1x1
    # thus, the output becomes (batch_size, 1, 1, 2048)
    x = GlobalAveragePooling2D()(x)

    # fully connected layer for intermediate features
    x = Dense(1024, activation='relu')(x)

    # dropout layer to prevent overfitting
    # 50% of the input will be dropped to zero
    x = Dropout(0.5)(x)

    # final output layer
    predictions = Dense(num_classes, activation='softmax')(x)

    # create the model
    model = Model(inputs=feature_extractor.input, outputs=predictions)
    
    # compile the model
    model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

    return model 

def create_resnet152v2_model(num_classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """
    Create a new ResNet152v2 model with transfer learning. 
    This model is based on the 2016 paper "Identity Mappings in Deep Residual Networks" by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
    Avilable at: https://arxiv.org/abs/1603.05027

    Args:
        weights (str, optional): Initial weights. Defaults to "imagenet".
        input_shape (tuple, optional): Shape of input images. Defaults to (224, 224, 3).

    Returns:
        tf.keras.application.ResNet152 : Model. 
    """

    # create a base model
    feature_extractor = ResNet152V2(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        input_shape=input_shape # specify input shape
    ) 

    # freeze the base model
    freeze_base_model(feature_extractor)

    x = feature_extractor.output 

    # change the spatial dimensions to 1x1
    # thus, the output becomes (batch_size, 1, 1, 2048)
    x = GlobalAveragePooling2D()(x)

    # fully connected layer for intermediate features
    x = Dense(1024, activation='relu')(x)

    # dropout layer to prevent overfitting
    # 50% of the input will be dropped to zero
    x = Dropout(0.5)(x)

    # final output layer
    predictions = Dense(num_classes, activation='softmax')(x)

    # create the model
    model = Model(inputs=feature_extractor.input, outputs=predictions)
    
    # compile the model
    model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

    return model    