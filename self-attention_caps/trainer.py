'''
File to train the neural network in various configurations.
By running this file, figures are saved in a folder.
'''

from datetime import date
import tensorflow as tf
from utils import Dataset, plotImages, plotWrongImages, plotHistory, Logger, plotter_from_df
from models import EfficientCapsNet
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme()


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def train_model(model_name=None, param_dict=None):
    '''
    Function that trains the model given the desired parameters, tests it and produces usefull plots as well as json data & logfiles.
    '''
    with open("config.json", "r") as fp:
        args = json.load(fp)
    ### CHANGE THE PARAMETERS HERE e.g. args["patates"] = 3  ###
    if param_dict is not None:
        # Overwrite the parameters in config.json with the ones in param_dict.
        for key, value in param_dict.items():
            args[key] = value
    with open("config.json", "w") as fp:
        json.dump(args, fp, indent=0)

    if 'model_name' in args and model_name is None:
        model_name = args['model_name']
        
    
    folder_name = os.path.join("./experiments",f"efficient_capsnetv3_{date.today().strftime('%d-%m-%y')}_dataset:{model_name}_batch:{args['batch_size']}_epochs:{args['epochs']}_lr:{args['lr']}_deconv:{args['deconvolution']}_multihead:{args['multihead']}_num_heads:{args['num_heads']}_algorithm_1_or_2:{args['algorithm_1_or_2']}_no_reconstructor:{args['no_reconstructor']}")
    custom_model_path = os.path.join(folder_name,"model_weights.h5")
    custom_tensorboard_path = os.path.join(folder_name,"board_bin")
    configuration_in_machine_readable_format_path = os.path.join(folder_name, "config.json")
    train_log_name = os.path.join(folder_name,"log_csv.log")
    log = Logger(f"logfile.logs", folder_name)
    with open(configuration_in_machine_readable_format_path, "w") as fp:
        fp.write(json.dumps(args, indent=0))
    log.info_message(f"Efficient Capsnet using {model_name} dataset. \n")
    log.info_message("Parameters of the training procedure. \n")
    log.print_train_args(args)

    dataset = Dataset(model_name, config_path='config.json')
    model_train = EfficientCapsNet(model_name, 'train', 'config.json', custom_model_path, custom_model_path, custom_tensorboard_path, verbose=True, no_reconstructor=args.get('no_reconstructor'))

    log.info_message("Loading dataset...")
    dataset_train, dataset_val = dataset.get_tf_data() 
    log.info_message("Dataset loaded...")

    # Start training
    history = model_train.train(dataset, initial_epoch=0, log=log, log_csv_name=train_log_name)

    history_dict = history.history
    # Save data to csv.
    history.history['Epoch'] = np.arange(1,args['epochs'] + 1)
    results = pd.DataFrame(history.history) 

    results.to_json(f"{os.path.join(folder_name, 'results.json')}", indent=1)

    log.info_message("Creating figures and saving lists of training data.")
    plotter_from_df(results, folder_name, args['no_reconstructor'])

    log.info_message("Finished training.")

    # Testing
    model_test = EfficientCapsNet(model_name, 'test', 'config.json', custom_model_path, custom_model_path, custom_tensorboard_path, True, no_reconstructor=args.get('no_reconstructor'))

    model_test.load_graph_weights() # load graph weights (bin folder unlsee custom path)

    # Test the model on the test set. It consists of the same images as the val set. However, the  transformations may be different.
    model_test.evaluate(dataset.X_test, dataset.y_test, log)

    log.info_message("Finished testing.")
    return
#TODO: Remember to check if the multihead attention is activated, the weights are stored. (ok)
#TODO: Parametrize more things (e.g. kernel size) (Maybe no point on doing that because there are sertain values that work -  can change them through the code itself) ok

# Start training MNIST dataset
# model_name = 'MNIST'
param_dict = {'epochs':2, 'multihead':1, 'original':0, 'batch_size':8, 'deconvolution':0, 'num_heads':2 , 'algorithm_1_or_2':'RooMAV'}

#train_model(model_name, param_dict)

train_model()

