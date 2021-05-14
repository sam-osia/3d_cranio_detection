from abc import ABCMeta, abstractmethod
import itertools
import random
import json
from subprocess import call

import tensorflow as tf
import tensorflow.keras as keras

import sys
sys.path.insert(0, '..')
from utils.utils import *


class BaseModel(metaclass=ABCMeta):
    def __init__(self, model_architecture, model_name, tag=None,
                 run_id=None,
                 hyperparams_range=None,
                 hyperparams=None):

        self.model_architecture = model_architecture
        self.model_name = model_name

        self.run_id = run_id

        if self.run_id is None:
            tag_ext = 'test'
        else:
            tag_ext = f'run_{run_id}'

        if tag is None:
            self.tag = tag_ext
        else:
            self.tag = f'{tag}_{tag_ext}'

        self.model_dir = f'./results/{self.model_architecture}/{self.model_name}'
        self.log_dir = os.path.join(self.model_dir, 'logs')
        self.run_name = time.strftime(f'{self.model_architecture}_{self.model_name}_{self.tag}_%Y_%m_%d-%H_%M_%S')
        self.run_dir = os.path.join(self.model_dir, self.run_name)

        if self.run_id is None:
            self.hyperparams = hyperparams
        else:
            self.hyperparams = self.generate_hyperparams(hyperparams_range)

        mkdir(self.model_dir)
        mkdir(self.log_dir)
        mkdir(self.run_dir)

        self.save_hyperparams(self.hyperparams)

    def generate_hyperparams(self, hyperparams_range):
        keys, vals = zip(*hyperparams_range.items())
        trials = [dict(zip(keys, v)) for v in itertools.product(*vals)]
        hyperparams = random.sample(trials, 1)[0]

        return hyperparams

    def save_hyperparams(self, hyperparams):
        with open(os.path.join(self.run_dir, f'hyperparams.json'), 'w') as f:
            json.dump(hyperparams, f)

    def parse_hyperparams(self):
        hyperparams = json.loads(os.path.join(self.run_dir, 'hyperparams.json'))
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

        logdir = get_log_dir(f'{self.model_dir}/tb_logs',
                             f'{self.model_architecture}_{self.model_name}_{self.tag}')

        savedir = get_save_dir(f'{self.run_dir}', 'weights')

        early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        tensorboard_cb = keras.callbacks.TensorBoard(logdir)

        model.fit(x_train, y_train,
                  validation_data=(x_valid, y_valid),
                  epochs=2,
                  callbacks=[early_stopping_cb, tensorboard_cb],
                  validation_split=0.2)

        print(model.evaluate(x_valid, y_valid))

        model.save(savedir)

    def submit_job(self):
        pass


if __name__ == '__main__':
    set_path()
