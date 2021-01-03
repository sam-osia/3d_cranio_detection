import sys
sys.path.insert(0, '..')
from utils.utils import *

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv3D, MaxPooling3D, Dropout
from tensorflow.keras.utils import to_categorical
import h5py

from model_base import BaseModel
import argparse


class Voxel3DModel(BaseModel):
    def __init__(self, tag=None, run_id=None):

        model_architecture = 'Voxel_3D'
        model_name = 'MNIST'

        hyperparams_range = {
            'n_conv_layers': [2, 3],
            'dropout': [0, 0.25, 0.5],
        }

        test_hyperparams = {
            'n_conv_layers': 2,
            'dropout': 0.4,
        }

        hyperparams = None
        if run_id is None:
            hyperparams = test_hyperparams

        super(Voxel3DModel, self).__init__(model_architecture, model_name, tag,
                                           run_id,
                                           hyperparams_range,
                                           hyperparams)

    def create_model(self, n_conv_layers, dropout) -> keras.models.Model:
        model = Sequential()
        model.add(Input(shape=(16, 16, 16, 1)))
        for i in range(n_conv_layers):
            model.add(Conv3D(32, (3, 3, 3), activation='relu'))
            model.add(MaxPooling3D(pool_size=(2, 2, 2)))
            model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))

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


def create_model(args):
    print('hello')
    model = Voxel3DModel(**vars(args))
    return model


if __name__ == '__main__':
    set_path()
    parser = argparse.ArgumentParser()
    parser.add_argument('-R', '--run_id', default=None)
    parser.add_argument('-T', '--tag', default=None)

    args = parser.parse_args()
    model = create_model(args)
    model.run_pipeline()
