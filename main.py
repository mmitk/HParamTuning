from __future__ import print_function
import sherpa
import io
import pandas as pd
import matplotlib
import numpy as np
from sherpa.algorithms import *
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import *
import time
import json
import os

LOSS_FUNCTIONS = {'mse': keras.losses.MeanSquaredError(),
                  'mae': keras.losses.MeanAbsoluteError(),
                  'mape': keras.losses.MeanAbsolutePercentageError(),
                  'msle': keras.losses.MeanSquaredLogarithmicError(),
                  'cos': keras.losses.CosineSimilarity(),
                  'huber': keras.losses.Huber(),
                  'logcosh': keras.losses.LogCosh()
                }

METRICS = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity','RootMeanSquaredError']

EPOCHS = 100

def get_data(train_file = 'data/train.cxv', test_file = 'data/test.cxv', val_file = 'data/val.cxv'):
    
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    val = pd.read_csv(val_file)

    train_x = train
    train_y = train_x[['Output 1', 'Output 2', 'Output 3']].copy()
    train_x = train_x.drop(['Output 1', 'Output 2', 'Output 3'], axis=1)

    test_x = test
    test_y = test_x[['Output 1', 'Output 2', 'Output 3']].copy()
    test_x = test_x.drop(['Output 1', 'Output 2', 'Output 3'], axis=1)

    val_x = val
    val_y = val_x[['Output 1', 'Output 2', 'Output 3']].copy()
    val_x = val_x.drop(['Output 1', 'Output 2', 'Output 3'], axis=1)

    return {'train_x':train_x, 'train_y':train_y, 'test_x':test_x, 'test_y':test_y, 'val_x': val_x, 'val_y': val_y, 'train':train, 'test':test, 'val': val}

def set_params_subset_one(loss_functions):
    hparams = [sherpa.Discrete('num_units', [12,26,32]),
           sherpa.Discrete('num_reg_units', [12,26,32]),
           sherpa.Choice('activation',['sigmoid','relu','linear','tanh']),
           sherpa.Continuous('learning_rate', [1e-4, 1e-2]),
           sherpa.Choice('loss_function',[lossname for lossname in loss_functions.keys()])]
    return hparams

def create_fit_model(train_x, train_y, val_x, val_y, test_x, test_y, study, trial, model_path, loss_functions):

    num_units = trial.parameters['num_units']
    num_reg_units = trial.parameters['num_reg_units']
    lr = trial.parameters['learning_rate']
    act = trial.parameters['activation']
    loss = trial.parameters['loss_function']

    model = tf.keras.models.Sequential()

    model.add(Dense(num_units,activation=act,input_shape = (len(train_x.keys()),)))
    model.add(Dense(num_reg_units, bias_regularizer=keras.regularizers.l2(1e-5), activity_regularizer=keras.regularizers.l2(1e-5)))
    model.add(tf.keras.layers.Dense(3, activation='linear'))

    model.compile(loss=loss_functions[loss], optimizer = tf.optimizers.Adam(learning_rate=lr), metrics=METRICS)

    for i in range(EPOCHS):
        model.fit(train_x,train_y, validation_data=(val_x,val_y))
        scores = model.evaluate(test_x,test_y)

        scores_dict = dict()
        for i, score in scores:
            scores_dict[model.metric_names[i]] = score

        rmse = scores_dict['root_mean_squared_error']
        scores_dict.pop('root_mean_squared_error',None)

        study.add_observation(trial=trial, iteration=i,objective=rmse, context=scores_dict)
        if study.should_trial_stop(trial):
            break
    
    model.save(model_path)


def run_study(algorithm, name, parameters, train_x, train_y, val_x, val_y, test_x, test_y):

    # create study
    study = sherpa.Study(parameters=parameters,
                     algorithm=algorithm,
                     lower_is_better=True)

    study_name = 'studies/{}'.format(name)
    session_num = 0
    for trial in study:
        model_path = 'models/{}/run{}'.format(name, session_num)
        create_fit_model(train_x, train_y, val_x, val_y, test_x, test_y, study, trial, model_path=model_path,loss_functions=LOSS_FUNCTIONS)
        study.finalize(trial=trial)
        session_num += 1
    
    # save study
    study.save(study_name)

if __name__ == '__main__':

    data_dict = get_data()

    algorithms_dict = {
                        'GPyOpt50':sherpa.algorithms.GPyOpt(max_num_trials = 50),
                        'SuccessHalving': sherpa.algorithms.SuccessiveHalving(),
                        'GridSearch' : sherpa.algorithms.GridSearch(),
                        'RandomSearch': sherpa.algorithms.RandomSearch()
                    }

    params = set_params_subset_one(LOSS_FUNCTIONS)

    time_dict = dict()

    for alg in algorithms_dict:
        start = time.time()

        run_study(algorithms_dict[alg], alg, params, data_dict['train_x'],data_dict['train_y'],data_dict['val_x'],data_dict['val_y'],data_dict['test_x'],data_dict['test_y'])

        end = time.time()

        exc_time = end - start

        time_dict[alg] = exc_time
    
    # save times dictionary
    fname = 'Execution{}'.format(str(len(os.listdir('times')+1)))