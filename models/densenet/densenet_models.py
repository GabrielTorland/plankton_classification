from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
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


def create_densenet121_model_no_fc(classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """Creates the DenseNet121 model.

    Returns:
        keras.engine.functional.Functional: The DenseNet121 model.
    """
    # Load the DenseNet121 model
    base_model = DenseNet121(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        input_shape=input_shape # specify input shape
    )

    # Add a global average pooling layer
    x = base_model.output
    
    x= GlobalAveragePooling2D()(x)
    
    # Add a classification layer
    predictions = Dense(classes, activation='softmax')(x)

    # Model to train
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
        

    
def create_densenet169_model_no_fc(classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """Creates the DenseNet169 model.

    Returns:
        keras.engine.functional.Functional: The DenseNet169 model.
    """
    # Load the DenseNet169 model
    base_model = DenseNet169(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        input_shape=input_shape # specify input shape
    )

    # Add a global average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Add a classification layer
    predictions = Dense(classes, activation='softmax')(x)

    # Model to train
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

    
def create_densenet201_model_no_fc(classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """Creates the DenseNet201 model.

    Returns:
        keras.engine.functional.Functional: The DenseNet201 model.
    """
    # Load the DenseNet201 model
    base_model = DenseNet201(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        input_shape=input_shape # specify input shape
    )

    # Add a global average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Add a classification layer
    predictions = Dense(classes, activation='softmax')(x)

    # Model to train
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
   

    
    
    
def create_densenet121_model(classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """Creates the DenseNet121 model.

    Returns:
        keras.engine.functional.Functional: The DenseNet121 model.
    """
    # Load the DenseNet121 model
    base_model = DenseNet121(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        input_shape=input_shape # specify input shape
    )

    # Freeze the base model
    freeze_base_model(base_model)

    # Add a global average pooling layer
    x = base_model.output
    
    x= GlobalAveragePooling2D()(x)
    
    # Batch normalization
    x = BatchNormalization()(x)
    
    # dropout layer to prevent overfitting
    # 50% of the input will be dropped to zero
    x= Dropout(0.5)(x)

    # Add two fully connected layer
    x= Dense(1024,activation='relu')(x)
    x= Dense(512,activation='relu')(x) 
    
    # dropout layer to prevent overfitting
    # 50% of the input will be dropped to zero
    x = Dropout(0.5)(x)
    
    # Batch normalization
    x = BatchNormalization()(x)

    # Add a classification layer
    predictions = Dense(classes, activation='softmax')(x)

    # Model to train
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def create_densenet169_model(classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """Creates the DenseNet169 model.

    Returns:
        keras.engine.functional.Functional: The DenseNet169 model.
    """
    # Load the DenseNet169 model
    base_model = DenseNet169(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        input_shape=input_shape # specify input shape
    )

    # Freeze the base model
    freeze_base_model(base_model)

    # Add a global average pooling layer
    x = base_model.output
    
    x= GlobalAveragePooling2D()(x)
    
    # Batch normalization
    x = BatchNormalization()(x)
    
    # dropout layer to prevent overfitting
    # 50% of the input will be dropped to zero
    x= Dropout(0.5)(x)

    # Add two fully connected layer
    x= Dense(1024,activation='relu')(x)
    x= Dense(512,activation='relu')(x) 
    
    # dropout layer to prevent overfitting
    # 50% of the input will be dropped to zero
    x = Dropout(0.5)(x)
    
    # Batch normalization
    x = BatchNormalization()(x)

    # Add a classification layer
    predictions = Dense(classes, activation='softmax')(x)

    # Model to train
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model



def create_densenet201_model(classes, weights=DEFAULT_WEIGHTS, input_shape=DEFAULT_INPUT_SHAPE):
    """Creates the DenseNet201 model.

    Returns:
        keras.engine.functional.Functional: The DenseNet201 model.
    """
    # Load the DenseNet201 model
    base_model = DenseNet201(
        include_top=False, # do not include the classification layer
        weights=weights, # load pre-trained weights
        input_shape=input_shape # specify input shape
    )

    # Freeze the base model
    freeze_base_model(base_model)

    # Add a global average pooling layer
    x = base_model.output
    
    x= GlobalAveragePooling2D()(x)
    
    # Batch normalization
    x = BatchNormalization()(x)
    
    # dropout layer to prevent overfitting
    # 50% of the input will be dropped to zero
    x= Dropout(0.5)(x)

    # Add two fully connected layer
    x= Dense(1024,activation='relu')(x)
    x= Dense(512,activation='relu')(x) 
    
    # dropout layer to prevent overfitting
    # 50% of the input will be dropped to zero
    x = Dropout(0.5)(x)
    
    # Batch normalization
    x = BatchNormalization()(x)

    # Add a classification layer
    predictions = Dense(classes, activation='softmax')(x)

    # Model to train
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model