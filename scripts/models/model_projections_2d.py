import sys

sys.path.insert(0, '..')
from utils.utils import *

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, \
    concatenate
from tensorflow.keras.utils import to_categorical
import h5py

from model_base import BaseModel
import argparse
import random

import matplotlib.pyplot as plt


class Projection2DModel(BaseModel):
    def __init__(self, tag=None, run_id=None):

        model_architecture = 'Projection_2D'
        model_name = 'Phone_tests'

        hyperparams_range = {
            'num_angles': [1, 3]
        }

        test_hyperparams = {
            'num_angles': 1
        }

        hyperparams = None
        if run_id is None:
            hyperparams = test_hyperparams

        super(Projection2DModel, self).__init__(model_architecture, model_name, tag,
                                                run_id,
                                                hyperparams_range,
                                                hyperparams)

    def create_model(self, num_angles) -> keras.models.Model:
        inputs = []
        layers = []

        for angle in range(num_angles):
            input = Input(shape=(128, 128, 1))
            layer = Conv2D(8, kernel_size=5, strides=2, activation='relu')(input)
            layer = BatchNormalization()(layer)
            layer = Conv2D(16, kernel_size=3, strides=1, activation='relu')(input)
            layer = BatchNormalization()(layer)
            layer = Conv2D(32, kernel_size=3, strides=1, activation='relu')(layer)
            layer = BatchNormalization()(layer)
            layer = Conv2D(64, kernel_size=3, strides=1, activation='relu')(layer)
            layer = BatchNormalization()(layer)
            layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(layer)

            layers.append(layer)
            inputs.append(input)

        if num_angles == 1:
            merged = layer
        else:
            merged = concatenate(layers)

        merged = Conv2D(256, kernel_size=3, strides=1, activation='relu', name='final_conv')(merged)
        merged = BatchNormalization()(merged)
        merged = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(merged)

        merged = Flatten()(merged)
        merged = Dense(512, activation='relu')(merged)
        merged = Dropout(0.5)(merged)
        merged = Dense(8, activation='relu')(merged)
        merged = Dropout(0.5)(merged)
        output = Dense(5, activation="softmax", name='predictions')(merged)

        model = keras.models.Model(inputs=inputs, outputs=output)
        print(model.summary())

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def load_data(self, **kwargs):
        total_samples = 1000
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        for i in range(total_samples):
            x = np.random.rand(128, 128, 1)
            y = np.array([0, 0, 1, 0, 0])
            if i < total_samples * 0.8:
                x_train.append(x)
                y_train.append(y)
            else:
                x_test.append(x)
                y_test.append(y)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        print(x_train.shape)

        return x_train, y_train, x_test, y_test


def create_model(args):
    print('hello')
    model = Projection2DModel(**vars(args))
    return model


def run_model_top_view():
    model = Projection2DModel(tag='playing_around').create_model(num_angles=1)
    model.load_weights('./results/Projection_2D/Pouria/3D_CNN_collapsed_top_view.h5')

    depth_files = ['./data/processed/voxel/black_no_cap/depths.npy',
                   './data/processed/voxel/black_with_cap/depths.npy',
                   './data/processed/voxel/white_no_cap/depths.npy',
                   './data/processed/voxel/white_with_cap/depths.npy']

    # depth_files = ['./data/processed/rgbd/s_1_top/depths.npy']

    depths_raw = []

    data = None
    for depth_file in depth_files:
        depth_data = np.load(depth_file)
        depths_raw.append(depth_data.copy())
        if data is None:
            data = depth_data.reshape(1, 128, 128, 1)
        else:
            data = np.concatenate((data, depth_data.reshape(1, 128, 128, 1)), axis=0)

    results = model.predict(data)

    for i, result in enumerate(results):
        print(f'{i} --> {result}')
        plt.subplot(len(results), 2, i * 2 + 1), plt.imshow(depths_raw[i]), plt.xticks([]), plt.yticks([]), plt.title(
            depth_files[i].split('/')[-2])
        plt.subplot(len(results), 2, i * 2 + 2), plt.pie(result,
                                                         labels=['Sagital', 'Metopic', 'Unicoronal', 'Plagiocephaly',
                                                                 'Normal'])
    plt.show()


def run_model_top_back_view():
    model = Projection2DModel(tag='two_views').create_model(num_angles=2)
    model.load_weights('./results/Projection_2D/Pouria/3D_CNN_collapsed_top_back_view.h5')
    img1 = np.random.rand(1, 128, 128, 1)
    img2 = np.random.rand(1, 128, 128, 1)
    result = model.predict([img1, img2])
    print(result)



if __name__ == '__main__':
    set_path()
    parser = argparse.ArgumentParser()
    parser.add_argument('-R', '--run_id', default=None)
    parser.add_argument('-T', '--tag', default=None)

    # args = parser.parse_args()
    # model = create_model(args)
    # model.run_pipeline()

    run_model_top_back_view()

