from keras.optimizers import Adam
from model import vgg16
import data


def train():
    x, y = data.collect_data()
    x = data.data_preprocessing(x,224)
    x_train, y_train, x_valid, y_valid = data.split_train_valid(x,y,0.75)
    epochs = 10
    batch_size = 10
    model = vgg16()
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train,
              batch_size=batch_size,
              epochs=epochs, validation_data=(x_valid, y_valid))
    model.save("model.h5")
    loss, accuracy =model.evaluate(x_valid,y_valid,batch_size=batch_size)
    print('loss:',loss)
    print('accuracy:', accuracy)

train()