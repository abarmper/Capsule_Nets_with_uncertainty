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
import os
import json
import numpy as np
import tensorflow as tf
from utils.layers import PrimaryCaps, FCCaps, Length
from utils.tools import get_callbacks, marginLoss, multiAccuracy
from utils.dataset import Dataset
from utils import pre_process_multimnist
from models import efficient_capsnet_graph_mnist, efficient_capsnet_graph_smallnorb, efficient_capsnet_graph_multimnist, original_capsnet_graph_mnist, efficient_capsnet_graph_cifar10
from tqdm.notebook import tqdm


class Model(object):
    """
    A class used to share common model functions and attributes.
    
    ...
    
    Attributes
    ----------
    model_name: str
        name of the model (Ex. 'MNIST')
    mode: str
        model modality (Ex. 'test')
    config_path: str
        path configuration file
    verbose: bool
    
    Methods
    -------
    load_config():
        load configuration file
    load_graph_weights():
        load network weights
    predict(dataset_test):
        use the model to predict dataset_test
    evaluate(X_test, y_test):
        comute accuracy and test error with the given dataset (X_test, y_test)
    save_graph_weights():
        save model weights
    """
    def __init__(self, model_name, mode='test', config_path='config.json', verbose=True, no_reconstructor=False):
        self.model_name = model_name
        self.model = None
        self.mode = mode
        self.config_path = config_path
        self.config = {}
        self.model_path = None
        self.verbose = verbose
        self.no_reconstructor = no_reconstructor
        self.load_config()


    def load_config(self):
        """
        Load config file
        """
        with open(self.config_path) as json_data_file:
            self.config = json.load(json_data_file)
    

    def load_graph_weights(self):
        try:
            self.model.load_weights(self.model_path)
        except Exception as e:
            print("[ERRROR] Graph Weights not found into dictionary: \n", self.config)   
        
        
    def predict(self, dataset_test):
        return self.model.predict(dataset_test)


    def evaluate(self, X_test, y_test, log):
        print('-'*30 + f'{self.model_name} Evaluation' + '-'*30)
        if self.model_name == "MULTIMNIST":
            acc = []
            if self.no_reconstructor:
                dataset_test = pre_process_multimnist.generate_tf_data_test_no_reconstructor(X_test, y_test, self.config["shift_multimnist"], n_multi=self.config['n_overlay_multimnist'])
                for X,y in tqdm(dataset_test,total=len(X_test)):
                    y_pred = self.model.predict(X)
                    acc.append(multiAccuracy(y, y_pred))
                acc = np.mean(acc)
            else:
                dataset_test = pre_process_multimnist.generate_tf_data_test(X_test, y_test, self.config["shift_multimnist"], n_multi=self.config['n_overlay_multimnist'])
                for X,y in tqdm(dataset_test,total=len(X_test)):
                    y_pred,X_gen1,X_gen2 = self.model.predict(X)
                    acc.append(multiAccuracy(y, y_pred))
                acc = np.mean(acc)
        else:
            if self.no_reconstructor:
                y_pred =  self.model.predict(X_test)
            else:
                y_pred, X_gen =  self.model.predict(X_test)
            acc = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
        test_error = 1 - acc
        log.info_message('Test acc: {}'.format(acc))
        log.info_message(f"Test error [%]: {(test_error):.4%}")
        if self.model_name == "MULTIMNIST":
            log.info_message(f"N° misclassified images: {int(test_error*len(y_test)*self.config['n_overlay_multimnist'])} out of {len(y_test)*self.config['n_overlay_multimnist']}")
        else:
            log.info_message(f"N° misclassified images: {int(test_error*len(y_test))} out of {len(y_test)}")


    def save_graph_weights(self):
        self.model.save_weights(self.model_path)



class EfficientCapsNet(Model):
    """
    A class used to manage an Efficiet-CapsNet model. 'model_name' and 'mode' define the particular architecure and modality of the generated network.
    
    ...
    
    Attributes
    ----------
    model_name: str
        name of the model (Ex. 'MNIST')
    mode: str
        model modality (Ex. 'test')
    config_path: str
        path configuration file
    custom_path: str
        custom weights path
    verbose: bool
    
    Methods
    -------
    load_graph():
        load the network graph given the model_name
    train(dataset, initial_epoch)
        train the constructed network with a given dataset. All train hyperparameters are defined in the configuration file

    """
    def __init__(self, model_name, mode='test', config_path='config.json', custom_save_path=None, custom_load_path=None, dustom_tb_path=None, verbose=True, no_reconstructor=False):
        Model.__init__(self, model_name, mode, config_path, verbose, no_reconstructor)

        if custom_load_path is not None:
            self.model_path = custom_load_path
        else:
            self.model_path = os.path.join(self.config['saved_model_dir'], f"efficient_capsnet_{self.model_name}.h5")

        if custom_save_path is not None:
            self.model_path_new_train = custom_save_path
        else:
            self.model_path_new_train = os.path.join(self.config['saved_model_dir'], f"efficient_capsnet{self.model_name}_new_train.h5")

        if dustom_tb_path is not None:
            self.tb_path = dustom_tb_path
        else:
            self.tb_path = os.path.join(self.config['tb_log_save_dir'], f"efficient_capsnet_{self.model_name}")
            
        if no_reconstructor is not None:
            self.no_reconstructor = no_reconstructor
        else:
            self.no_reconstructor = False

        self.load_graph()
    

    def load_graph(self):
        if self.model_name == 'MNIST':
            self.model = efficient_capsnet_graph_mnist.build_graph(self.config['MNIST_INPUT_SHAPE'], self.mode, self.verbose, self.config['deconvolution'], self.config['multihead'], self.config["original"], self.config["num_heads"], self.config["algorithm_1_or_2"], self.config["scale_the_embedding"], self.config["use_agreement_criterion"], self.no_reconstructor)
        elif self.model_name == 'SMALLNORB':
            self.model = efficient_capsnet_graph_smallnorb.build_graph(self.config['SMALLNORB_INPUT_SHAPE'], self.mode, self.verbose, self.config['deconvolution'], self.config['multihead'], self.config["original"], self.config["num_heads"], self.config["algorithm_1_or_2"], self.config["scale_the_embedding"], self.config["use_agreement_criterion"], self.no_reconstructor)
        elif self.model_name == 'MULTIMNIST':
            self.model = efficient_capsnet_graph_multimnist.build_graph(self.config['MULTIMNIST_INPUT_SHAPE'], self.mode, self.verbose, self.config['deconvolution'], self.config['multihead'], self.config["original"], self.config["num_heads"], self.config["algorithm_1_or_2"], self.config["scale_the_embedding"], self.config["use_agreement_criterion"], self.no_reconstructor)
        elif self.model_name == 'CIFAR10':
            self.model = efficient_capsnet_graph_cifar10.build_graph(self.config['CIFAR10_INPUT_SHAPE'], self.mode, self.verbose, self.config['deconvolution'], self.config['multihead'], self.config["original"], self.config["num_heads"], self.config["algorithm_1_or_2"], self.config["scale_the_embedding"], self.config["use_agreement_criterion"], self.no_reconstructor)
            
    def train(self, dataset=None, initial_epoch=0, log=None, log_csv_name=None):
        callbacks = get_callbacks(self.tb_path, self.model_path_new_train, self.config['lr_dec'], self.config['lr'], log, log_csv_name, no_reconstruction=self.no_reconstructor)

        if dataset is None:
            dataset = Dataset(self.model_name, self.config_path)
        if self.no_reconstructor:
            dataset_train, dataset_val = dataset.get_tf_data_no_reconstructor() 
        else:
            dataset_train, dataset_val = dataset.get_tf_data()    
        if self.no_reconstructor:
            # No reconstructor is used.
            if self.model_name == 'MULTIMNIST':
                self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
                loss=[marginLoss],
                metrics={'Efficient_CapsNet': multiAccuracy})
                steps = 10*int(dataset.y_train.shape[0] / self.config['batch_size'])
            else:
                self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
                loss=[marginLoss],
                metrics={'Efficient_CapsNet': 'accuracy'})
                steps=None
        else:
            # Use reconstruction (also called decoder).
            if self.model_name == 'MULTIMNIST':
                self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
                loss=[marginLoss, 'mse', 'mse'],
                loss_weights=[1., self.config['lmd_gen']/2,self.config['lmd_gen']/2],
                metrics={'Efficient_CapsNet': multiAccuracy})
                steps = 10*int(dataset.y_train.shape[0] / self.config['batch_size'])
            else:
                self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
                loss=[marginLoss, 'mse'],
                loss_weights=[1., self.config['lmd_gen']],
                metrics={'Efficient_CapsNet': 'accuracy'})
                steps=None

        print('-'*30 + f'{self.model_name} train' + '-'*30)
        # print("DATAAAASET: ", dataset_train)
        history = self.model.fit((dataset_train),
          epochs=self.config['epochs'], steps_per_epoch=steps,
          validation_data=(dataset_val), batch_size=self.config['batch_size'], initial_epoch=initial_epoch,
          callbacks=callbacks)
        
        return history

            
        
        
class CapsNet(Model):
    """
    A class used to manage the original CapsNet architecture.
    
    ...
    
    Attributes
    ----------
    model_name: str
        name of the model (only MNIST provided)
    mode: str
        model modality (Ex. 'test')
    config_path: str
        path configuration file
    verbose: bool
    n_routing: int
        number of routing interations
    
    Methods
    -------
    load_graph():
        load the network graph given the model_name
    train():
        train the constructed network with a given dataset. All train hyperparameters are defined in the configuration file
    """
    def __init__(self, model_name, mode='test', config_path='config.json', custom_path=None, verbose=True, n_routing=3):
        Model.__init__(self, model_name, mode, config_path, verbose)   
        self.n_routing = n_routing
        self.load_config()
        if custom_path != None:
            self.model_path = custom_path
        else:
            self.model_path = os.path.join(self.config['saved_model_dir'], f"efficient_capsnet_{self.model_name}.h5")
        self.model_path_new_train = os.path.join(self.config['saved_model_dir'], f"original_capsnet_{self.model_name}_new_train.h5")
        self.tb_path = os.path.join(self.config['tb_log_save_dir'], f"original_capsnet_{self.model_name}")
        self.load_graph()

    
    def load_graph(self):
        self.model = original_capsnet_graph_mnist.build_graph(self.config['MNIST_INPUT_SHAPE'], self.mode, self.n_routing, self.verbose)
        
    def train(self, dataset=None, initial_epoch=0):
        callbacks = get_callbacks(self.tb_path, self.model_path_new_train, self.config['lr_dec'], self.config['lr'])
        
        if dataset == None:
            dataset = Dataset(self.model_name, self.config_path)          
        dataset_train, dataset_val = dataset.get_tf_data()   


        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
              loss=[marginLoss, 'mse'],
              loss_weights=[1., self.config['lmd_gen']],
              metrics={'Original_CapsNet': 'accuracy'})

        print('-'*30 + f'{self.model_name} train' + '-'*30)

        history = self.model.fit(dataset_train,
          epochs=self.config['epochs'],
          validation_data=(dataset_val), batch_size=self.config['batch_size'], initial_epoch=initial_epoch,
          callbacks=callbacks)
        
        return history
