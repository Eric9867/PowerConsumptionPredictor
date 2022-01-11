import tensorflow as tf
import keras
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from data_processing import read_data

ZONES = ['Zone 1 Power Consumption', 'Zone 2 Power Consumption', 'Zone 3 Power Consumption']

def get_model_data(inputs, train_split=0.75, filepath='tetuan_city_power_consumption.csv'):
    data, zpcs = read_data(filepath)
    train_num = int(data.shape[0] * train_split)

    randomized_data = data[inputs + ZONES].sample(n=data.shape[0])

    xy_train = randomized_data[:train_num]
    xy_test = randomized_data[train_num:]

    def get_axis_data(df, inputs):
        x_axis = df[inputs]
        y_axis = df[ZONES]
        return np.asarray(x_axis).astype('float32'), np.asarray(y_axis).astype('float32')

    x_train, y_train = get_axis_data(xy_train, inputs)
    x_test, y_test = get_axis_data(xy_test, inputs)

    return x_train, y_train, x_test, y_test

def create_model(in_shape):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=in_shape))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(3))
    model.compile(
        optimizer='RMSprop',
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy']
    )
    return model


def train(model, x_train, y_train, plot_learning=False):

    history = model.fit(
        x=x_train, 
        y=y_train, 
        batch_size=2048,
        epochs=15, 
        callbacks=None,
        validation_split=0.05,
        verbose=False,
        shuffle=True)


    if plot_learning:
        history_dict = history.history
        acc = history_dict['accuracy']
        val_acc_values = history_dict['val_accuracy']
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, loss_values, 'bo', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    

def train_generic(
    model,
    inputs=['Month', 'Day', 'Time', 'Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows'], 
    batch_size=2048,
    save_weights=False):

    x_train, y_train, x_test, y_test = get_model_data(inputs)

    train(model, x_train, y_train, save_weights)
    model.evaluate(x=x_test, y=y_test, batch_size=batch_size)

    predictions = np.asarray(model.predict(x_test))

    r2_1 = r2_score(y_test[:, 0], predictions[:, 0])
    r2_2 = r2_score(y_test[:, 1], predictions[:, 1])
    r2_3 = r2_score(y_test[:, 2], predictions[:, 2])

    print(r2_1, r2_2, r2_3, sep='\n')
    return model, r2_1, r2_2, r2_3




if __name__ == '__main__' :
    # inputs = ['Month', 'Day', 'Time', 'Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows']
    # train_generic(model=create_model((len(inputs),)))

    print(create_model((8,)).summary())
