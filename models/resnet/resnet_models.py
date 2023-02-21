from tensorflow.keras import Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.models import Sequential
from tensorflow.image import resize


DEFAULT_INPUT_SHAPE = (224, 224, 3)
DEFAULT_WEIGHTS = "imagenet"


def freeze_base_model(model):
    """ Freeze the base model.

    Args:
        model (tensorflow.keras.applications.*): Base model. 
    """    
    model.trainable = False

def add_layers(base_model, classes):
    """ Adds trainable layers to the base model. 

    Args:
        base_model (tensorflow.keras.applications.*): Base model. 
        classes (_type_):  Number of classes. 
    Returns:
        tensorflow.keras.models.Sequential: ResNet model with trainable layers.
    """   
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(classes, activation="softmax"))

    return model


def create_resnet50_model(classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """
    Create a new ResNet50 model with transfer learning.
    This model is based on the 2015 paper "Deep Residual Learning for Image Recognition" by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
    Avilable at: https://arxiv.org/abs/1512.03385

    Args:
        weights (str, optional): Initial weights. Defaults to "imagenet".
        input_shape (tuple, optional): Shape of input images. Defaults to (256, 256, 3).

    Returns:
        tf.keras.application.ResNet50 : Base model. 
    """

    # create a base model
    base_model = ResNet50(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        input_shape=input_shape # specify input shape
    ) 

    # freeze the base model
    freeze_base_model(base_model)

    # add trainable layers
    model = add_layers(base_model, classes) 

    return model


def create_resnet50v2_model(classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """
    Create a new ResNet50v2 model with transfer learning. 
    This model is based on the 2016 paper "Identity Mappings in Deep Residual Networks" by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
    Avilable at: https://arxiv.org/abs/1603.05027

    Args:
        weights (str, optional): Initial weights. Defaults to "imagenet".
        input_shape (tuple, optional): Shape of input images. Defaults to (256, 256, 3).

    Returns:
        tf.keras.application.ResNet50v2 : Base model. 
    """

    # create a base model
    base_model = ResNet50V2(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        input_shape=input_shape # specify input shape
    ) 

    # freeze the base model
    freeze_base_model(base_model)

    # add trainable layers
    model = add_layers(base_model, classes) 

    return model
    
def create_resnet101_model(classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """
    Create a new ResNet101 model with transfer learning. 
    This model is based on the 2015 paper "Deep Residual Learning for Image Recognition" by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
    Avilable at: https://arxiv.org/abs/1512.03385

    Args:
        weights (str, optional): Initial weights. Defaults to "imagenet".
        input_shape (tuple, optional): Shape of input images. Defaults to (256, 256, 3).

    Returns:
        tf.keras.application.ResNet101 : Base model. 
    """

    # create a base model
    base_model = ResNet101(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        input_shape=input_shape # specify input shape
    ) 

    # freeze the base model
    freeze_base_model(base_model)

    # add trainable layers
    model = add_layers(base_model, classes) 

    return model

def create_resnet101v2_model(classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """
    Create a new ResNet101v2 model with transfer learning. 
    This model is based on the 2016 paper "Identity Mappings in Deep Residual Networks" by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
    Avilable at: https://arxiv.org/abs/1603.05027

    Args:
        weights (str, optional): Initial weights. Defaults to "imagenet".
        input_shape (tuple, optional): Shape of input images. Defaults to (256, 256, 3).

    Returns:
        tf.keras.application.ResNet50 : Base model. 
    """

    # create a base model
    base_model = ResNet101V2(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        input_shape=input_shape # specify input shape
    ) 

    # freeze the base model
    freeze_base_model(base_model)

    # add trainable layers
    model = add_layers(base_model, classes) 

    return model 

def create_resnet152_model(classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """
    Create a new ResNet152 model with transfer learning. 
    This model is based on the 2015 paper "Deep Residual Learning for Image Recognition" by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
    Avilable at: https://arxiv.org/abs/1512.03385

    Args:
        weights (str, optional): Initial weights. Defaults to "imagenet".
        input_shape (tuple, optional): Shape of input images. Defaults to (256, 256, 3).

    Returns:
        tf.keras.application.ResNet152 : Base model. 
    """

    # create a base model
    base_model = ResNet152(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        input_shape=input_shape # specify input shape
    ) 

    # freeze the base model
    freeze_base_model(base_model)

    # add trainable layers
    model = add_layers(base_model, classes) 

    return model 

def create_resnet152v2_model(classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """
    Create a new ResNet152v2 model with transfer learning. 
    This model is based on the 2016 paper "Identity Mappings in Deep Residual Networks" by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
    Avilable at: https://arxiv.org/abs/1603.05027

    Args:
        weights (str, optional): Initial weights. Defaults to "imagenet".
        input_shape (tuple, optional): Shape of input images. Defaults to (256, 256, 3).

    Returns:
        tf.keras.application.ResNet152 : Base model. 
    """

    # create a base model
    base_model = ResNet152V2(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        input_shape=input_shape # specify input shape
    ) 

    # freeze the base model
    freeze_base_model(base_model)

    # add trainable layers
    model = add_layers(base_model, classes) 

    return model    