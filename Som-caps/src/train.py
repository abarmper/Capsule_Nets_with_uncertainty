from distutils.log import Log
import tensorflow as tf
from data.dataset import Dataset
from network.model import MyModel, som_capsnet_graph, build_graph, MULTIMNIST_build_graph, SMALLNORB_build_graph
import numpy as np
from utils.tools import Logger, get_callbacks, save_history_and_plots, get_callbacks_eval, multiAccuracy, marginLoss, MarginLoss, MultiAccuracy
from utils.visuals import plotHistory
import argparse
from datetime import date
import tensorflow as tf
import os
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# tf.compat.v1.disable_eager_execution()
tf.config.run_functions_eagerly(True)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', type=str, default="MNIST",
                    help='choose between MNIST (default) or Fashion-MNIST or CIFAR10 or SMALLNORB or MULTIMNIST')
parser.add_argument('--dataset_file', type=str, default=None,
                    help='set the directory where the dataset is/will be located (default: ~/.keras/datasets/)')
parser.add_argument('--thetas', type=str, default='1.0',
                    help='Neighborhood function, list of numbers separated by commas e.g. "1.0, 0.2, 0.1". It should be a series of decimals, separated be comma. The first digit is the output of the neighbourhood function when distance=0, the second when distance=1 etc. (default: "1.0")')
parser.add_argument('--batch_size', type=int, default=16, metavar='Theta',
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=2, metavar='Eps',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--rooting_iterations', help='# of routing iterations. After training, you can even set iterations to zero! (default: 1)', type=int, default=1)
parser.add_argument('--with_softmax' , action='store_true', default=False, help='If set then instead of calculating the winners using argmax, we use softmax. Of course, then there is no much point in iterating over many times and thetas should be 1.0. (default: False)')
parser.add_argument('--with_reconstruction', action='store_true', default=False, help='If set, we use reconstruction regularization. (default: False)')
parser.add_argument('--with_deconvolution' , action='store_true', default=False, help='If we use reconstruction, there is this option which if set, we use an advanced deconvolution-based reconstructor. (default: False)')
parser.add_argument('--small', action='store_true', default=False, help='If we set small, number of primary capsules is greatly reduced. (default: False)')
parser.add_argument('--non_reduced_votes', action='store_true', default=False, help='If we set non_reduced_votes, then there are no different transformation matrices for each digit capsule. In other words, if we set this argument, all digit caps will have the same view of the primary capsules (just copied n^(L+1) times). (default: False)')
parser.add_argument('--transformation_matrices_for_each_capsule', type=int, default=None, metavar='nTMat',
                    help='if non_reduced_votes is set, then we need to define the number of transformation_matrices_for_each_capsule (default: number of digit caps)')
parser.add_argument('--with_elaborate_slow_logger', action='store_true', default=False)
parser.add_argument('--lr_som', type=float, default=1.0, metavar='LRsom',
                    help='learning rate for SOM (default: 1.0)')
parser.add_argument('--radical', action='store_true', default=False, help='If we set radical, then the differences in SOM algorithm are computed as the new votes u. This is an extreme case that differs from original SOM significantly. (default: False)')
parser.add_argument('--norm_type', type=int, default=0, metavar='Type_norm',
                    help='If set to 0, squash is taken as normalizing function. If set to 1, then we devide each vector with it\'s own length so as to make it unit vector (we can do that without affecting the output as we do not use length to find the prediction). (default: 0)')
parser.add_argument('--normalize_d_in_loop', action='store_true', default=False, help='If we set normalize_d_in_loop, then we normalize the digit caps at each iteration within the loop. (default: False)')
parser.add_argument('--normalize_digit_caps', action='store_true', default=False, help='If we set normalize_digit_caps, then we normalize the digit caps at the end of the loop. (default: False)')
parser.add_argument('--normalize_votes', action='store_true', default=False, help='If we set normalize_votes, then we normalize the votes before we enter the loop. (default: False)')
parser.add_argument('--take_into_account_similarity', action='store_true', default=False, help='If we set take_into_account_similarity, then we scale the digit capsules with the similarity scores we found when computing som (similarity is measured as a product between the votes and the digit caps). (default: False)')
parser.add_argument('--take_into_account_winner_ratios', action='store_true', default=False, help='If we set take_into_account_winner_ratios, then we scale the digit capsules with the winner ratios. If a digit capsule has many capsules that send their votes to him, the digit capsule will be boosted (stronger digit caps vector and similarity). (default: False)')
parser.add_argument('--tanh_like', action='store_true', default=False, help='If we set tanh_like, then we normalize the vectors using tanh instead of softmax. Its either softmax, or tanh or hardwinners (do not set softmax and tanh for that, its the default). So you can not set both tanh and softmax. Also, if you set softmax or tanh, then iterations should be one (or even zero). (default: False)')


args = parser.parse_args()



if args.with_reconstruction:
    gen=True
else:
    gen=False

if args.with_deconvolution:
    dec = True
else:
    dec = False

if args.dataset == 'CIFAR10':
    if_cifar_Set_true = True
else:
    if_cifar_Set_true = False

small = True if args.small else False
reduced_votes = False if args.non_reduced_votes else True
softmax = True if args.with_softmax else False
radical = True if args.radical else False
normalize_d_in_loop = True if args.normalize_d_in_loop else False
normalize_digit_caps = True if args.normalize_digit_caps else False
normalize_votes = True if args.normalize_votes else False
take_into_account_similarity = True if args.take_into_account_similarity else False
take_into_account_winner_ratios = True if args.take_into_account_winner_ratios else False
tanh_like = True if args.tanh_like else False
with_elaborate_slow_logger = True if args.with_elaborate_slow_logger else False


folder_name = os.path.join("../experiments",f"SOM_capsnet_{date.today().strftime('%d_%m_%y')}_dataset:{args.dataset}_batch:{args.batch_size}_epochs:{args.epochs}_lr:{args.lr}_recon:{gen}_with_deconvolution:{dec}_with_elaborate_slow_logger:{with_elaborate_slow_logger}_rooting_iterations:{args.rooting_iterations}_thetas:{args.thetas}_with_softmax:{softmax}_small:{small}_radical:{radical}_tanhlike:{tanh_like}")
custom_model_path = os.path.join(folder_name,"model_weights.h5")
custom_tensorboard_path = os.path.join(folder_name,"board_bin")
log_path = os.path.join(folder_name, "my_csv_logfile.log")
log_path_ev = os.path.join(folder_name, "my_csv_logfile_for_testing.log")
log = Logger(f"logfile.logs", folder_name)
log.info_message(f"Efficient Capsnet using {args.dataset} dataset. \n")
log.info_message("Parameters of the training procedure. \n")
log.print_train_args(args)

# Convert list of strings to list of floats.
thetas_list = list(map(float, args.thetas.split(',')))

log.info_message("Load dataset... \n")
ds = Dataset(args.dataset, args.batch_size, gen)
log.info_message("Dataset loaded. \n")

# Set number of transformation matrices.
transformation_matrices_for_each_capsule = len(ds.class_names) if args.transformation_matrices_for_each_capsule is None else args.transformation_matrices_for_each_capsule

if gen:
    input_shape = ds.ds_train.element_spec[0][0].shape
else:
    input_shape = ds.ds_train.element_spec[0].shape # Input shape (None, 28, 28, 1) for MNIST (without batch size).

# Parse input thetas:
thetas = [float(theta) for theta in args.thetas.split(',')]


#model = MyModel()
#model = som_capsnet_graph(input_shape[1:])
if args.dataset != 'SMALLNORB' and args.dataset != 'MULTIMNIST':
    model = build_graph(input_shape[1:], batch_size=args.batch_size, digit_vector_size=8, gen=gen, num_classes=10, softmax=softmax, neighboor_thetas=thetas, iterations=args.rooting_iterations, reduced_votes=reduced_votes, training=True, deco=dec, cifar=if_cifar_Set_true, small=small, transformation_matrices_for_each_capsule=transformation_matrices_for_each_capsule, 
                        lr_som =args.lr_som, radical=radical, normalize_d_in_loop=normalize_d_in_loop,
                        normalize_digit_caps=normalize_digit_caps, normalize_votes=normalize_votes, norm_type=args.norm_type, take_into_account_similarity=take_into_account_similarity, take_into_account_winner_ratios=take_into_account_winner_ratios, tanh_like=tanh_like)
    model.build(input_shape)
    model.summary(print_fn=log.info_message)
elif args.dataset == 'SMALLNORB':
    model = SMALLNORB_build_graph(input_shape[1:], batch_size=args.batch_size, digit_vector_size=8, gen=gen, num_classes=5, softmax=softmax, neighboor_thetas=thetas, iterations=args.rooting_iterations, reduced_votes=reduced_votes, training=True, deco=dec, cifar=if_cifar_Set_true, small=small, transformation_matrices_for_each_capsule=transformation_matrices_for_each_capsule, 
                        lr_som =args.lr_som, radical=radical, normalize_d_in_loop=normalize_d_in_loop,
                        normalize_digit_caps=normalize_digit_caps, normalize_votes=normalize_votes, norm_type=args.norm_type, take_into_account_similarity=take_into_account_similarity, take_into_account_winner_ratios=take_into_account_winner_ratios, tanh_like=tanh_like)
    model.build(input_shape)
    model.summary(print_fn=log.info_message)
elif args.dataset == 'MULTIMNIST':
    model = MULTIMNIST_build_graph(input_shape[1:], batch_size=args.batch_size, digit_vector_size=8, gen=gen, num_classes=10, softmax=softmax, neighboor_thetas=thetas, iterations=args.rooting_iterations, reduced_votes=reduced_votes, training=True, deco=dec, cifar=if_cifar_Set_true, small=small, transformation_matrices_for_each_capsule=transformation_matrices_for_each_capsule, 
                        lr_som =args.lr_som, radical=radical, normalize_d_in_loop=normalize_d_in_loop,
                        normalize_digit_caps=normalize_digit_caps, normalize_votes=normalize_votes, norm_type=args.norm_type, take_into_account_similarity=take_into_account_similarity, take_into_account_winner_ratios=take_into_account_winner_ratios, tanh_like=tanh_like)
    model.build(input_shape)
    model.summary(print_fn=log.info_message)

# model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],optimizer=tf.keras.optimizers.Adam(args.lr))

# If you want to run eagerly, replace ds.ds_train with (ds.X_train, ds.y_train) and ds.ds_test with (ds.X_test, ds.y_test).
# Also, in all files comment out tf.compat.v1.disable_eager_execution() and set tf.config.run_functions_eagerly() to True (by typing tf.config.run_functions_eagerly(True)).
if with_elaborate_slow_logger:
    logger_callback = log
else:
    logger_callback = None
    
if args.dataset != 'SMALLNORB' and args.dataset != 'MULTIMNIST':
    ext = 0
elif args.dataset == 'SMALLNORB':
    ext = 1
else: # args.dataset == 'MULTIMNIST':
    ext = 2


if not gen:
    if args.dataset != 'SMALLNORB' and args.dataset != 'MULTIMNIST':
        margin_loss = MarginLoss(sparce=False, num_classes=10)
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],optimizer=tf.keras.optimizers.Nadam(args.lr))
        history = model.fit(ds.ds_train, epochs=args.epochs, validation_data=ds.ds_val, steps_per_epoch=None, callbacks=get_callbacks(custom_tensorboard_path, custom_model_path, logger_callback, log_path))
    elif args.dataset == 'SMALLNORB':
        margin_loss = MarginLoss(sparce=True, num_classes=5)
        model.compile(loss=[margin_loss],
        metrics=[tf.keras.metrics.CategoricalAccuracy()],optimizer=tf.keras.optimizers.Nadam(args.lr))
        history = model.fit(ds.ds_train, epochs=args.epochs, validation_data=ds.ds_val, steps_per_epoch=None, callbacks=get_callbacks(custom_tensorboard_path, custom_model_path, logger_callback, log_path))
    elif args.dataset == 'MULTIMNIST':
        margin_loss = MarginLoss(sparce=True, num_classes=10)
        model.compile(loss=[margin_loss],
        metrics=[MultiAccuracy()],optimizer=tf.keras.optimizers.Nadam(args.lr))
        history = model.fit(ds.ds_train, epochs=args.epochs, validation_data=ds.ds_val, steps_per_epoch=10*int(ds.y_train.shape[0] / args.batch_size), validation_steps=10*int(ds.y_test.shape[0] / args.batch_size), callbacks=get_callbacks(custom_tensorboard_path, custom_model_path, logger_callback, log_path))
else: 
    if args.dataset != 'SMALLNORB' and args.dataset != 'MULTIMNIST':
        margin_loss = MarginLoss(sparce=False, num_classes=10)
        model.compile(loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'mse'], loss_weights=[0.9, 0.1],
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],optimizer=tf.keras.optimizers.Nadam(args.lr))
        history = model.fit(ds.ds_train, epochs=args.epochs, validation_data=ds.ds_val, steps_per_epoch=None, callbacks=get_callbacks(custom_tensorboard_path, custom_model_path, logger_callback, log_path))
    elif args.dataset == 'SMALLNORB':
        margin_loss = MarginLoss(sparce=True, num_classes=5)
        model.compile(loss=[margin_loss, 'mse'], loss_weights=[0.9, 0.1],
        metrics=[tf.keras.metrics.CategoricalAccuracy()],optimizer=tf.keras.optimizers.Nadam(args.lr))
        history = model.fit(ds.ds_train, epochs=args.epochs, validation_data=ds.ds_val, steps_per_epoch=None, callbacks=get_callbacks(custom_tensorboard_path, custom_model_path, logger_callback, log_path))
    elif args.dataset == 'MULTIMNIST':
        margin_loss = MarginLoss(sparce=True, num_classes=10)
        model.compile(loss=[margin_loss, 'mse', 'mse'], loss_weights=[0.8, 0.1, 0.1],
        metrics=[MultiAccuracy()],optimizer=tf.keras.optimizers.Nadam(args.lr))
        history = model.fit(ds.ds_train, epochs=args.epochs, validation_data=ds.ds_val, steps_per_epoch=10*int(ds.y_train.shape[0] / args.batch_size), validation_steps=10*int(ds.y_test.shape[0] / args.batch_size), callbacks=get_callbacks(custom_tensorboard_path, custom_model_path, logger_callback, log_path))

save_history_and_plots(history, folder_name, args.epochs, gen, extended=ext)

# Test the model: build another one with training = False and load weights of the best model
log.info_message("\n\nTraining completed! Starting Testing.\n")

if args.dataset != 'SMALLNORB' and args.dataset != 'MULTIMNIST':
    model = build_graph(input_shape[1:], batch_size=args.batch_size, digit_vector_size=8, gen=gen, num_classes=10, softmax=softmax, neighboor_thetas=thetas, iterations=args.rooting_iterations, reduced_votes=reduced_votes, training=False, deco=dec, cifar=if_cifar_Set_true, small=small, transformation_matrices_for_each_capsule=transformation_matrices_for_each_capsule, 
                        lr_som =args.lr_som, radical=radical, normalize_d_in_loop=normalize_d_in_loop,
                        normalize_digit_caps=normalize_digit_caps, normalize_votes=normalize_votes, norm_type=args.norm_type, take_into_account_similarity=take_into_account_similarity, take_into_account_winner_ratios=take_into_account_winner_ratios, tanh_like=tanh_like)
    model.build(input_shape)
    model.summary(print_fn=log.info_message)
elif args.dataset == 'SMALLNORB':
    model = SMALLNORB_build_graph(input_shape[1:], batch_size=args.batch_size, digit_vector_size=8, gen=gen, num_classes=5, softmax=softmax, neighboor_thetas=thetas, iterations=args.rooting_iterations, reduced_votes=reduced_votes, training=False, deco=dec, cifar=if_cifar_Set_true, small=small, transformation_matrices_for_each_capsule=transformation_matrices_for_each_capsule, 
                        lr_som =args.lr_som, radical=radical, normalize_d_in_loop=normalize_d_in_loop,
                        normalize_digit_caps=normalize_digit_caps, normalize_votes=normalize_votes, norm_type=args.norm_type, take_into_account_similarity=take_into_account_similarity, take_into_account_winner_ratios=take_into_account_winner_ratios, tanh_like=tanh_like)
    model.build(input_shape)
    model.summary(print_fn=log.info_message)
elif args.dataset == 'MULTIMNIST':
    model = MULTIMNIST_build_graph(input_shape[1:], batch_size=args.batch_size, digit_vector_size=8, gen=gen, num_classes=10, softmax=softmax, neighboor_thetas=thetas, iterations=args.rooting_iterations, reduced_votes=reduced_votes, training=Faalse, deco=dec, cifar=if_cifar_Set_true, small=small, transformation_matrices_for_each_capsule=transformation_matrices_for_each_capsule, 
                        lr_som =args.lr_som, radical=radical, normalize_d_in_loop=normalize_d_in_loop,
                        normalize_digit_caps=normalize_digit_caps, normalize_votes=normalize_votes, norm_type=args.norm_type, take_into_account_similarity=take_into_account_similarity, take_into_account_winner_ratios=take_into_account_winner_ratios, tanh_like=tanh_like)
    model.build(input_shape)
    model.summary(print_fn=log.info_message)

log.info_message("\nLoad best weights...\n")
model.load_weights(custom_model_path)
log.info_message("Weights Loaded.\n")
log.info_message("\n Start evaluation.\n")


if not gen:
    if args.dataset != 'SMALLNORB' and args.dataset != 'MULTIMNIST':
        margin_loss = MarginLoss(sparce=False, num_classes=10)
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],optimizer=tf.keras.optimizers.Nadam(args.lr))
        history = model.evaluate(ds.ds_test, callbacks=get_callbacks_eval(log, log_path_ev))
        log.info_message("Evaluated the model and got these results: Total_loss {}, Accuracy: {}\n".format(history[0], history[1]))
    elif args.dataset == 'SMALLNORB':
        margin_loss = MarginLoss(sparce=True, num_classes=5)
        model.compile(loss=[margin_loss],
        metrics=[tf.keras.metrics.CategoricalAccuracy()],optimizer=tf.keras.optimizers.Nadam(args.lr))
        history = model.evaluate(ds.ds_test, callbacks=get_callbacks_eval(log, log_path_ev))
        log.info_message("Evaluated the model and got these results: ", history)
    elif args.dataset == 'MULTIMNIST':
        margin_loss = MarginLoss(sparce=True, num_classes=10)
        model.compile(loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
        metrics=[MultiAccuracy()],optimizer=tf.keras.optimizers.Nadam(args.lr))
        history = model.evaluate(ds.ds_test, callbacks=get_callbacks_eval(log, log_path_ev))
        log.info_message("Evaluated the model and got these results: ", history)
else: 
    if args.dataset != 'SMALLNORB' and args.dataset != 'MULTIMNIST':
        margin_loss = MarginLoss(sparce=False, num_classes=10)
        model.compile(loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 'mse'], loss_weights=[0.9, 0.1],
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],optimizer=tf.keras.optimizers.Nadam(args.lr))
        history = model.evaluate(ds.ds_test, callbacks=get_callbacks_eval(log, log_path_ev))
        log.info_message("Evaluated the model and got these results: Partial_loss somnet: {}, Partial_loss_generator: {}, Total_loss {}, Accuracy: {}, Gen_accuracy {}\n".format(history[0], history[1], history[2], history[3], history[4]))

    elif args.dataset == 'SMALLNORB':
        margin_loss = MarginLoss(sparce=True, num_classes=5)
        model.compile(loss=[margin_loss, 'mse'], loss_weights=[0.9, 0.1],
        metrics=[tf.keras.metrics.CategoricalAccuracy()],optimizer=tf.keras.optimizers.Nadam(args.lr))
        history = model.evaluate(ds.ds_test, callbacks=get_callbacks_eval(log, log_path_ev))
        log.info_message("Evaluated the model and got these results: ", history)
    elif args.dataset == 'MULTIMNIST':
        margin_loss = MarginLoss(sparce=True, num_classes=10)
        model.compile(loss=[margin_loss, 'mse', 'mse'], loss_weights=[0.8, 0.1, 0.1],
        metrics=[MultiAccuracy()],optimizer=tf.keras.optimizers.Nadam(args.lr))
        history = model.evaluate(ds.ds_test, callbacks=get_callbacks_eval(log, log_path_ev), steps = 10*int(ds.y_test.shape[0] / args.batch_size))
        log.info_message("Evaluated the model and got these results: ", history)


log.info_message("End of evaluation. All nominal.")
