from abc import ABCMeta, abstractmethod
import itertools
import random
import json

import tensorflow as tf
import tensorflow.keras as keras

import sys
sys.path.insert(0, '..')
from utils.utils import *


class BaseModel(metaclass=ABCMeta):
    def __init__(self, model_architecture, model_name, model_tag=None,
                 run_id=None, data_path=None,
                 hyperparams_range=None, generate_hyperparams=False,
                 hyperparams=None,
                 test_run=True):

        self.model_architecture = model_architecture
        self.model_name = model_name

        self.run_id = run_id
        self.data_path = data_path
        self.model_tag = model_tag

        self.model_dir = f'./results/{self.model_architecture}/{self.model_name}'
        self.hyperparams_dir = os.path.join(self.model_dir, 'hyperparameters')
        self.weights_dir = os.path.join(self.model_dir, 'model_weights')

        if generate_hyperparams:
            self.generate_hyperparams(hyperparams_range)

        if test_run:
            self.hyperparams = hyperparams
        else:
            self.hyperparams = self.parse_hyperparams()

        mkdir(self.model_dir)
        mkdir(self.hyperparams_dir)
        mkdir(self.weights_dir)

    def generate_hyperparams(self, hyperparams_range, num_trials):
        keys, vals = zip(*hyperparams_range.items())
        trials = [dict(zip(keys, v)) for v in itertools.product(*vals)]

        trial_params = random.sample(trials, num_trials)

        params_dir = os.path.join(self.model_dir, 'hyperparameters')
        mkdir(params_dir)

        for i, trial_param in enumerate(trial_params):
            with open(os.path.join(params_dir, f'run{i}.json'), 'w') as f:
                json.dump(trial_param, f)
        return

    def parse_hyperparams(self):
        hyperparams = json.loads(os.path.join(self.hyperparams_dir, f'run{self.run_id}.json'))
        return hyperparams

    @abstractmethod
    def create_model(self, **kwargs) -> keras.models.Model:
        pass

    @abstractmethod
    def load_data(self, **kwargs):
        x_train = []
        y_train = []
        x_valid = []
        y_valid = []
        return x_train, y_train, x_valid, y_valid

    def run_pipeline(self):
        x_train, y_train, x_valid, y_valid = self.load_data(**self.hyperparams)

        model = self.create_model(**self.hyperparams)

        logdir = get_log_dir(f'{self.model_dir}/tb_logs', self.model_architecture + f'_{self.model_tag}')
        savedir = get_save_dir(f'{self.model_dir}/model_weights', self.model_architecture + f'_{self.model_tag}')

        early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        tensorboard_cb = keras.callbacks.TensorBoard(logdir)

        model.fit(x_train, y_train,
                  validation_data=(x_valid, y_valid),
                  epochs=1000,
                  callbacks=[early_stopping_cb, tensorboard_cb],
                  validation_split=0.2)

        print(savedir)
        model.save(savedir)


if __name__ == '__main__':
    set_path()
