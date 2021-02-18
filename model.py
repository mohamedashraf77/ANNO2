from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
from keras.engine import Model
from keras.optimizers import Adam

def vgg16():
    # load model
    input_shape = (224, 224, 3)
    model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in model.layers:
        layer.trainable = False

    last = model.layers[-1].output
    x = Flatten()(last)
    x = Dense(1000, activation='relu', name='fc1')(x)
    x = Dense(1, activation='sigmoid', name='fc2')(x)
    my_model = Model(model.input, x)
    my_model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return(my_model)