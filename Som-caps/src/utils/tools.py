'''
This file contains vvarious custom functions used in the other files.
To make the other code files cleaner, we move the declaration of the tools in this file.

Created by Alexandros Barmperis
'''

import tensorflow as tf
import logging
import os
from matplotlib import pyplot as plt
from utils.visuals import plotter_from_df, plotter_from_df_smallnorb, plotter_from_df_multimnist
import numpy as np
import pandas as pd


def squash(s: tf.float32) ->tf.float32:
    '''
    Squash activation function.
    ...
    
    Inputs:
    s: tensor
    '''
    n = tf.norm(s, axis=-1,keepdims=True)
    return tf.multiply(n**2/(1+n**2)/(n + tf.keras.backend.epsilon()), s)


def tanh_vector(s: tf.float32) -> tf.float32:
    '''
    tanh function for vectors
    '''
    n = tf.norm(s, axis=-1,keepdims=True)
    return tf.multiply(tf.math.divide(tf.math.tanh(n),n), s) # tanh for vectors

class Log_Callback(tf.keras.callbacks.Callback):
    def __init__(self, tf_logger):
        super().__init__()
        self.tf_logger = tf_logger

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        self.tf_logger.info_message("Starting training.")

    def on_train_end(self, logs=None):
        self.tf_logger.info_message("Stop training; got the following results: {}".format(logs))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        self.tf_logger.info_message("Start epoch {} of training.".format(epoch))

    def on_epoch_end(self, epoch, logs=None):
        self.tf_logger.info_message("End epoch {} of training; got the following results: {}".format(epoch, logs))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        self.tf_logger.info_message("Start testing.")

    def on_test_end(self, logs=None):
        self.tf_logger.info_message("Stop testing; got log: {}".format(logs))

    # def on_train_batch_end(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     self.tf_logger.info_message("...Training: end of batch {}; got results: {}".format(batch, logs))

class Logger():
    """
    This class is responsible for logging information about the training process as well as keeping 
    the results and creating graghs for the losses.
    """
    def __init__(self, filename, path):
        # Define the format in which log messeges will apear.
        FILE_LOG_FORMAT = "%(asctime)s %(filename)s:%(lineno)d %(message)s"
        CONSOLE_LOG_FORMAT = "%(levelname)s %(message)s"

        if not os.path.isdir(path):
            # Create a folder containing the experiments.
            try:
                os.makedirs(path)
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                print ("Successfully created the directory %s " % path)

        log_file = os.path.join(path, filename)


        if not os.path.isfile(log_file):
            open(log_file, "w").close()

        logging.basicConfig(level=logging.INFO, format=CONSOLE_LOG_FORMAT)
        self.logger = logging.getLogger()

        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(FILE_LOG_FORMAT)
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)
        

    def info_message(self, message):
        self.logger.info(message)
        return

    def print_train_args(self, args):
        for arg in vars(args):
            message = str(arg) + ": " + str(getattr(args, arg))
            self.logger.info(message)


def learn_scheduler(lr_dec=0.0001, lr=0.001):
    def learning_scheduler_fn(epoch):
        lr_new = lr * (lr_dec ** epoch)
        return lr_new if lr_new >= 5e-5 else 5e-5
    return learning_scheduler_fn

def get_callbacks(tb_log_save_path, saved_model_path, log, log_path='my_csv_log.log'):
    tb = tf.keras.callbacks.TensorBoard(log_dir=tb_log_save_path, histogram_freq=0)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(saved_model_path, monitor='val_loss',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    # lr_decay = tf.keras.callbacks.LearningRateScheduler(learn_scheduler())

    #reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_CapsNet_accuracy', factor=0.9,
    #                          patience=4, min_lr=0.00001, min_delta=0.0001, mode='max')

    if log is not None:
        my_logger = Log_Callback(log)
        return [tb, model_checkpoint, my_logger]
    else:
        csv_logger = tf.keras.callbacks.CSVLogger(log_path)
        return [tb, model_checkpoint, csv_logger]
    
def get_callbacks_eval(log, log_path='my_csv_log.log'):

    if log is not None:
        my_logger = Log_Callback(log)
        return [my_logger]
    else:
        csv_logger = tf.keras.callbacks.CSVLogger(log_path)
        return [csv_logger]

def save_history_and_plots(history, folder_name_to_save, eps, gen, extended=0):
    history_dict = history.history
    # Save data to csv.
    history.history['Epoch'] = np.arange(1,eps + 1)
    results = pd.DataFrame(history.history) 
    results.to_json(f"{os.path.join(folder_name_to_save, 'results.json')}", indent=1)    
    # Plot data and save figures. 
    if extended==0:
        plotter_from_df(results, folder_name_to_save, gen)
    elif extended ==1: #smallnorb
        plotter_from_df_smallnorb(results, folder_name_to_save, gen)
    else: # extended==2 (multimnist)
        plotter_from_df_multimnist(results, folder_name_to_save, gen)
                
    return

def multiAccuracy(y_true, y_pred):
    label_pred = tf.argsort(y_pred,axis=-1)[:,-2:]
    label_true = tf.argsort(y_true,axis=-1)[:,-2:]
    
    acc = tf.reduce_sum(tf.cast(label_pred[:,:1]==label_true,tf.int8),axis=-1) + \
          tf.reduce_sum(tf.cast(label_pred[:,1:]==label_true,tf.int8),axis=-1)
    acc /= 2
    return tf.reduce_mean(acc,axis=-1)

class MultiAccuracy(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        self.acc = 0.0
        super().__init__(**kwargs)
    def update_state(self, y_true, y_pred, sample_weight=None):
        # print("\n\ny_true shape::::::",y_true.shape)
        label_pred = tf.argsort(y_pred,axis=-1)[:,-2:]
        label_true = tf.argsort(y_true,axis=-1)[:,-2:]
        
        self.acc = tf.reduce_sum(tf.cast(label_pred[:,:1]==label_true,tf.int8),axis=-1) + \
            tf.reduce_sum(tf.cast(label_pred[:,1:]==label_true,tf.int8),axis=-1)
        self.acc = self.acc  / 2
        return 
    def result(self):
        return self.acc
    def get_config(self):
        return super().get_config()

def marginLoss(y_true, y_pred):
    lbd = 0.5
    m_plus = 0.9
    m_minus = 0.1
    
    L = y_true * tf.square(tf.maximum(0., m_plus - y_pred)) + \
    lbd * (1 - y_true) * tf.square(tf.maximum(0., y_pred - m_minus))

    return tf.reduce_mean(tf.reduce_sum(L, axis=1))

class MarginLoss(tf.keras.losses.Loss):
    '''
    Custom Margin Loss.
    sparce argument is about how y_true is given in call:
    If y_true is one_hote then set sparce to True.
    '''
    def __init__(self, lbd=0.5, m_plus=0.9, m_minus = 0.1, sparce = False, num_classes=10, **kwargs):
        self.lbd = lbd
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.sparce = sparce
        self.num_classes = num_classes
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        # print("\n\n shape y_true: ",y_true.shape, y_true.dtype, "\nshape y_pred: ", y_pred.shape)
        if not self.sparce:
            y_true = tf.keras.backend.one_hot(tf.cast(y_true, dtype=tf.int32), num_classes=self.num_classes)
        L = y_true * tf.square(tf.maximum(0., self.m_plus - y_pred)) + \
        self.lbd * (1 - y_true) * tf.square(tf.maximum(0., y_pred - self.m_minus))

        return tf.reduce_mean(tf.reduce_sum(L, axis=1))
    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "lbd": self.lbd,
                "m_plus": self.m_plus,
                "m_minus":self.m_minus}