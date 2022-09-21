from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os


def plotHistory(history):
    """
    Plot the loss and accuracy curves for training and validation 
    """
    pd.DataFrame(history.history).plot(figsize=(8, 5), y=list(history.history.keys())[0:-1:2])
    plt.grid(True)
    plt.show()


def plotter_from_df(df, folder_name, gen):
    if gen:
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
        plt.plot(df.Epoch, df.Som_CapsNet_partial_loss, label='train loss of caps')
        plt.plot(df.Epoch, df.val_Som_CapsNet_partial_loss, label='val loss of caps')
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
        plt.plot(df.Epoch, df.Som_CapsNet_partial_sparse_categorical_accuracy, label='train acc')
        plt.plot(df.Epoch, df.val_Som_CapsNet_partial_sparse_categorical_accuracy, label='val acc')
        plt.legend(loc='upper right')
        plt.savefig(f"{os.path.join(folder_name, 'train_acc.png')}")
        
    else:
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
        plt.title('Train & validation accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.plot(df.Epoch, df.sparse_categorical_accuracy, label='train acc')
        plt.plot(df.Epoch, df.val_sparse_categorical_accuracy, label='val acc')
        plt.legend(loc='upper right')
        plt.savefig(f"{os.path.join(folder_name, 'train_acc.png')}")
    return


def plotter_from_df_multimnist(df, folder_name, gen):
    if gen:
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
        plt.plot(df.Epoch, df.Som_CapsNet_partial_loss, label='train loss of caps')
        plt.plot(df.Epoch, df.val_Som_CapsNet_partial_loss, label='val loss of caps')
        plt.legend(loc='upper right')
        plt.savefig(f"{os.path.join(folder_name, 'loss_only_capsnet.png')}")

        fig, ax = plt.subplots(figsize=(8, 8))
        # plot losses
        plt.title('Loss of Reconstruction')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.plot(df.Epoch, df.Generator_loss, label='train loss of generator 1')
        plt.plot(df.Epoch, df.Generator_1_loss, label='train loss of generator 2')
        plt.plot(df.Epoch, df.val_Generator_loss, label='val loss of generator 1')
        plt.plot(df.Epoch, df.val_Generator_1_loss, label='val loss of generator 2')
        plt.legend(loc='upper right')
        plt.savefig(f"{os.path.join(folder_name, 'loss_only_reconstruction.png')}")

        fig, ax = plt.subplots(figsize=(8, 8))
        # plot losses
        plt.title('Train & validation accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.plot(df.Epoch, df.Som_CapsNet_partial_multi_accuracy, label='train acc')
        plt.plot(df.Epoch, df.val_Som_CapsNet_partial_multi_accuracy, label='val acc')
        plt.legend(loc='upper right')
        plt.savefig(f"{os.path.join(folder_name, 'train_acc.png')}")
        
    else:
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
        plt.title('Train & validation accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.plot(df.Epoch, df.categorical_multi_accuracy, label='train acc')
        plt.plot(df.Epoch, df.val_categorical_multi_accuracy, label='val acc')
        plt.legend(loc='upper right')
        plt.savefig(f"{os.path.join(folder_name, 'train_acc.png')}")
    return


def plotter_from_df_smallnorb(df, folder_name, gen):
    if gen:
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
        plt.plot(df.Epoch, df.Som_CapsNet_partial_loss, label='train loss of caps')
        plt.plot(df.Epoch, df.val_Som_CapsNet_partial_loss, label='val loss of caps')
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
        plt.plot(df.Epoch, df.Som_CapsNet_partial_categorical_accuracy, label='train acc')
        plt.plot(df.Epoch, df.val_Som_CapsNet_partial_categorical_accuracy, label='val acc')
        plt.legend(loc='upper right')
        plt.savefig(f"{os.path.join(folder_name, 'train_acc.png')}")
        
    else:
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
        plt.title('Train & validation accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.plot(df.Epoch, df.categorical_accuracy, label='train acc')
        plt.plot(df.Epoch, df.val_categorical_accuracy, label='val acc')
        plt.legend(loc='upper right')
        plt.savefig(f"{os.path.join(folder_name, 'train_acc.png')}")
    return