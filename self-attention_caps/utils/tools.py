# Copyright 2021 Vittorio Mazzia & Francesco Salvetti. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme()


def learn_scheduler(lr_dec, lr):
    def learning_scheduler_fn(epoch):
        lr_new = lr * (lr_dec ** epoch)
        return lr_new if lr_new >= 5e-5 else 5e-5
    return learning_scheduler_fn

class CustomCallback(tf.keras.callbacks.Callback):
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

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        self.tf_logger.info_message("...Training: end of batch {}; got results: {}".format(batch, logs))


def get_callbacks(tb_log_save_path, saved_model_path, lr_dec, lr, log):
    tb = tf.keras.callbacks.TensorBoard(log_dir=tb_log_save_path, histogram_freq=0)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(saved_model_path, monitor='val_Efficient_CapsNet_accuracy',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    lr_decay = tf.keras.callbacks.LearningRateScheduler(learn_scheduler(lr_dec, lr))

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_CapsNet_accuracy', factor=0.9,
                              patience=4, min_lr=0.00001, min_delta=0.0001, mode='max')

    if log != None:
        my_logger = CustomCallback(log)
        return [tb, model_checkpoint, lr_decay, my_logger]
    else:
        return [tb, model_checkpoint, lr_decay]


def marginLoss(y_true, y_pred):
    lbd = 0.5
    m_plus = 0.9
    m_minus = 0.1
    
    L = y_true * tf.square(tf.maximum(0., m_plus - y_pred)) + \
    lbd * (1 - y_true) * tf.square(tf.maximum(0., y_pred - m_minus))

    return tf.reduce_mean(tf.reduce_sum(L, axis=1))


def multiAccuracy(y_true, y_pred):
    label_pred = tf.argsort(y_pred,axis=-1)[:,-2:]
    label_true = tf.argsort(y_true,axis=-1)[:,-2:]
    
    acc = tf.reduce_sum(tf.cast(label_pred[:,:1]==label_true,tf.int8),axis=-1) + \
          tf.reduce_sum(tf.cast(label_pred[:,1:]==label_true,tf.int8),axis=-1)
    acc /= 2
    return tf.reduce_mean(acc,axis=-1)

def plotter_from_df(df, folder_name):

    fig, ax = plt.subplots(figsize=(8, 8))
    # plot losses
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(df.Epoch, df.loss, label='train loss')
    plt.plot(df.Epoch, df.val_loss, label='val loss')
    plt.legend(loc='upper right')
    plt.savefig(f"{os.path.join(folder_name, 'loss.png')}")

    fig, ax = plt.subplots(figsize=(8, 8))
    # plot losses
    plt.title('Loss of Capsnet only')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(df.Epoch, df.Efficient_CapsNet_loss, label='train loss of caps')
    plt.plot(df.Epoch, df.val_Efficient_CapsNet_loss, label='val loss of caps')
    plt.legend(loc='upper right')
    plt.savefig(f"{os.path.join(folder_name, 'loss_only_capsnet.png')}")

    fig, ax = plt.subplots(figsize=(8, 8))
    # plot losses
    plt.title('Loss of Reconstruction')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(df.Epoch, df.Generator_loss, label='train loss of generator')
    plt.plot(df.Epoch, df.val_Generator_loss, label='val loss of generator')
    plt.legend(loc='upper right')
    plt.savefig(f"{os.path.join(folder_name, 'loss_only_reconstruction.png')}")

    fig, ax = plt.subplots(figsize=(8, 8))
    # plot losses
    plt.title('Train & validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(df.Epoch, df.Efficient_CapsNet_accuracy, label='train acc')
    plt.plot(df.Epoch, df.val_Efficient_CapsNet_accuracy, label='val acc')
    plt.legend(loc='upper right')
    plt.savefig(f"{os.path.join(folder_name, 'train_acc.png')}")
    return

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

    def print_train_args(self, arg_dict):
        for key,value in arg_dict.items():
            message = key + ": " + str(value) + "\n"
            self.logger.info(message)


