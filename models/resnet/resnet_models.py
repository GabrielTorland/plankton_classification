from tensorflow.keras import Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50


def freeze_base_model(model):
    # freeze the base model
    model.trainable = False


def create_resnet50_model(weights="imagenet", input_shape=(256, 256, 3)):
    """
    Create a new ResNet50 model with transfer learning.

    Args:
        weights (str, optional): Initial weights. Defaults to "imagenet".
        input_shape (tuple, optional): Shape of input images. Defaults to (256, 256, 3).

    Returns:
        tf.keras.application.ResNet50 : Base model. 
    """

    def create_base_model(weights="imagenet", input_shape=(256, 256, 3)):
        """
        Create a base model for transfer learning.

        Args:
            weights (str, optional): Initial weights. Defaults to "imagenet".
            input_shape (tuple, optional): Shape of input images. Defaults to (256, 256, 3).

        Returns:
            tf.keras.application.ResNet50 : Base model. 
        """    
        return ResNet50(
            include_top=False, # do not include the classification layer
            weights="imagenet", # load pre-trained weights
            input_shape=(256, 256, 3) # specify input shape
    )
    
    # create a base model
    base_model = create_base_model(weights=weights, input_shape=input_shape)

    # freeze the base model
    freeze_base_model(base_model)

    # create a new model
    inputs = Input(shape=(256, 256, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(inputs, x)

    return model
