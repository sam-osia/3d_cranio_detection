import h5py
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv3D, MaxPooling3D, Dropout
from tensorflow.keras.utils import to_categorical
import h5py
import numpy as np
import matplotlib.pyplot as plt

from model_base import BaseModel

import sys
sys.path.insert(0, '..')
from utils.utils import *


class Voxel3DModel(BaseModel):
    def __init__(self, model_tag=None,
                 run_id=None, data_path=None,
                 hyperparams_range=None, generate_hyperparams=False,
                 hyperparams=None,
                 test_run=True):

        model_architecture = 'Voxel 3D'
        model_name = 'MNIST test'

        super(Voxel3DModel, self).__init__(model_architecture, model_name, model_tag,
                                           run_id, data_path,
                                           hyperparams_range, generate_hyperparams,
                                           hyperparams,
                                           test_run)

    def create_model(self, n_conv_layers, dropout, n_class) -> keras.models.Model:
        model = Sequential()
        model.add(Input(shape=(16, 16, 16, 1)))
        for i in range(n_conv_layers):
            model.add(Conv3D(32, (3, 3, 3), activation='relu'))
            model.add(MaxPooling3D(pool_size=(2, 2, 2)))
            model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(n_class, activation='softmax'))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print(model.summary())

        return model

    def load_data(self, **kwargs):
        with h5py.File('./data/examples/3D_mnist/full_dataset_vectors.h5', 'r') as hf:
            # Split the data into training/test features/targets
            x_train = hf["X_train"][:]
            y_train = hf["y_train"][:]
            x_test = hf["X_test"][:]
            y_test = hf["y_test"][:]

            # Reshape data into 3D format
            x_train = x_train.reshape(-1, 16, 16, 16, 1)
            x_test = x_test.reshape(-1, 16, 16, 16, 1)

            # Convert target vectors to categorical targets
            y_train = to_categorical(y_train).astype(np.integer)
            y_test = to_categorical(y_test).astype(np.integer)

        return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    set_path()

    hyperparams = {
        'n_conv_layers': 2,
        'dropout': 0.4,
        'n_class': 10
    }

    model = Voxel3DModel(test_run=True, hyperparams=hyperparams)

    model.run_pipeline()
