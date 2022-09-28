"""CapsNet with MNIST

This code implements Capsule Network as it is presented in the paper "Dynamic Routing Between Capsules"
by Sasabour, Frosst and  Hinton. I will try to comment it step by step so as to better understand this implementation.
"""

from __future__ import print_function
from numpy.core.numeric import Inf

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from smallnorb import SmallNORB

import logging
import os
from datetime import date

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
sns.set_theme()

def print_learning_curve(train_loss, test_loss, eps):
    """
    Plot diagnostic learning curves.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.suptitle('Training Curves')
    # plot losses
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.plot(np.arange(1, eps + 1, step=1), train_loss, color='blue', label='train')
    plt.plot(np.arange(1, eps + 1, step=1), test_loss, color='orange', label='test')
    plt.legend(loc='upper right')
    return plt

def print_accuracy_curve(test_acc, eps):
    """
    Plot diagnostic accuracy curve.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.suptitle('Test Accuracy')
    # plot accuracy
    plt.title('Classification Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.plot(np.arange(1, eps + 1, step=1), test_acc, color='blue', label='train')

    return plt

def print_accuracy_curve_norm(test_acc, eps):
    """
    Plot diagnostic accuracy curve but with stnandard y axis from 0% to 100%.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.suptitle('Test Accuracy')
    # plot accuracy
    plt.title('Classification Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    ax.set_ylim((0, 100))
    plt.plot(np.arange(1, eps + 1, step=1), test_acc, color='blue', label='train')

    return plt

def squash(x: torch.Tensor) -> torch.Tensor:
    """Function that implements the squashing described in eq.1 in the paper.

    Parameters
    ----------
    x : torch.Tensor
        Input vector s_j that will be squashed.
        first dim: batch size
        second dim: number of capsules in the layer
        third dim: size of the output vector for each capsule.

    Returns
    ----------
    x : torch.Tensor
        squashed vector v_j
    """
    lengths2 = x.pow(2).sum(dim=2)  # vector s_j is stored in the third dimension of x.
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x


class AgreementRouting(nn.Module):
    """
    The Routing algorithm: an iterative routing-by-agreement mechanism by witch a lower level capsule
    layer sends it's output to higher level capsules whose activity vectors have a big scalar product
    with the prediction coming from the lower-level capsule.

    Parameters
    ----------
    input_caps : int
    output_caps : int
    n_iterations : int
    argmax : bool

    Methods
    -------
    forward
    """
    def __init__(self, input_caps, output_caps, n_iterations, argmax=False, arg_max_ones=True):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        self.argmax = argmax # If true, we compute parent vector as thechild vector with the maximul c_i.
        # Parameters aare like tensors but when assigned as Module attribiutes they are added to the parameters list.
        # Parameters go to cuda if model is on cuda.
        self.arg_max_ones = arg_max_ones # Only used if argmax is true. If arg_max_ones is true then the winning coefficient becomes one.
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps))) 
        

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()
        c = F.softmax(self.b, dim=1)
        s1 = (c.unsqueeze(2) * u_predict)
        s2 = s1.sum(dim=1)
        v = squash(s2)

        if self.n_iterations > 0:

            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            for r in range(self.n_iterations - 1):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = F.softmax(b_batch.view(-1, output_caps), dim=1).view(-1, input_caps, output_caps, 1)
                s1 = (c * u_predict)
                s2 = s1.sum(dim=1)
                v = squash(s2)
                
            if self.argmax:
                # Instead of using a weighted sum of votes, choose the vote of child capsule with the largest c coefficient and pass that vote only to the next layer (digit caps).

                c_zero = torch.zeros_like(c)
                values, idx = c.max(dim=1)
                if self.arg_max_ones:
                    c_ones = torch.ones_like(c)
                    c_sparse = c_zero.scatter(1, idx.unsqueeze(dim=1), c_ones) # Sparse patrix has zeros everywhere but the winning child capsule (for each parent capsule).
                    v = (c_sparse * u_predict).sum(dim=1)
                else:
                    c_sparse = c_zero.scatter(1, idx.unsqueeze(dim=1), values.unsqueeze(dim=1))
                    s = (c_sparse * u_predict).sum(dim=1)
                    v = squash(s) # This may not be needed bc s < 1 True for all but squash also pushes towords 1 at a sertain range.
        elif self.n_iterations < 0:
            # If iterations is < 0 then find the longest vote vector for each parent capsule and pass this one.
            u_c = u_predict.norm(dim=3) # Take the l2 norm of each vote vector to find which is the longest.
            values, idx = u_c.max(dim=1) 
            c_zero = torch.zeros_like(u_c)
            c_ones = torch.ones_like(u_c)
            c_sparse = c_zero.scatter(1, idx.unsqueeze(dim=1), c_ones) # Sparse patrix has zeros everywhere but the winning child capsule (for each parent capsule).
            v = (c_sparse.unsqueeze(dim=3) * u_predict).sum(dim=1)
        else:
            # This means that n_iterations ==0, we have already taken into account this case.
            pass

        return v, s1


class CapsLayer(nn.Module):
    """
    Capsule layer implementation. It stores the transformation matrices that are being learned during training.
    It also calls the routing algorithm at forward for computing the v_j's given the inputs u_i's.
    It connects the PrimaryCapsLayer and the DigitCaps layer.
    """
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim))
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the transformation matrices and initialises them with samples from uniform distribution.
        """
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        caps_output = caps_output.unsqueeze(2)
        u_predict = caps_output.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)
        v, s1 = self.routing_module(u_predict)
        return v, s1


class PrimaryCapsLayer(nn.Module):
    """
    The primary capsule layer (actually, the convolutional part, without the transformation matrices).

    Attribiutes
    -----------
    input_channels : int 
                    The number of channels of the input of the capsule layer.
    output_caps : int
                    Number of capsules channels in the primary capsule layer. In the paper this number is 32.
                    This is equivallent to the number of capsule layers and not the number of capsules in the
                    primaryCaps (wich is 32*6*6).
    output_dim : int
                    The dimensionality of the capsules output vector.
    """
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride):
        super(PrimaryCapsLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        out = self.conv(input)
        N, C, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)  # Group the output vectors together.

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(out.size(0), -1, out.size(4))
        out = squash(out)
        return out


class CapsNet(nn.Module):
    """
    Putting it all together. It encapsulates the network found in Figure 1 of the paper.
    """
    def __init__(self, routing_iterations, n_classes=10, input_channels=1, types_of_primary_caps=32, argmax=False, arg_max_ones=True):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=9, stride=1)
        self.primaryCaps = PrimaryCapsLayer(256, types_of_primary_caps, 8, kernel_size=9, stride=2)  # outputs 6*6
        self.num_primaryCaps = types_of_primary_caps * 6 * 6
        routing_module = AgreementRouting(self.num_primaryCaps, n_classes, routing_iterations, argmax, arg_max_ones)
        self.digitCaps = CapsLayer(self.num_primaryCaps, 8, n_classes, 16, routing_module)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.primaryCaps(x)
        x, s1 = self.digitCaps(x)
        probs = x.pow(2).sum(dim=2).sqrt()
        return x, probs, s1


class ReconstructionNet(nn.Module):
    """
    Decoder network. Acts as regularizer.
    """
    def __init__(self, n_dim=16, n_classes=10, in_channels=1, out_hw = 28):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim * n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, in_channels * out_hw * out_hw) # 28 == hight == width
        self.n_dim = n_dim
        self.n_classes = n_classes

    def forward(self, x, target):
        # Mask so as to feed the decoder network with the DigitCaps vector prediction of the correct digit.
        # Dose the same apply during test time? (Where we do not have the target.)
        # Shouldn't we use the argmax() over the norms of digitCaps vectors during test time instead ? That's what we did.
        mask = Variable(torch.zeros((x.size()[0], self.n_classes)), requires_grad=False)

        if next(self.parameters()).is_cuda:
            mask = mask.cuda()
        if self.training:
            mask.scatter_(1, target.view(-1, 1), 1.)  # To one hot vector
        else:
            mask.scatter_(1, torch.argmax(torch.argmax(torch.norm(x, dim=2), dim=1, keepdim=False).view(-1, 1), dim=1, keepdim=True), 1.)  # To one hot vector
        mask = mask.unsqueeze(2)
        x = x * mask
        x = x.view(-1, self.n_dim * self.n_classes)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class CapsNetWithReconstruction(nn.Module):
    """
    CapsNet + ReconstructionNet : (Encoder + Decoder)
    """
    def __init__(self, capsnet, reconstruction_net):
        super(CapsNetWithReconstruction, self).__init__()
        self.capsnet = capsnet
        self.reconstruction_net = reconstruction_net

    def forward(self, x, target):
        x, probs, s1 = self.capsnet(x)
        reconstruction = self.reconstruction_net(x, target)
        return reconstruction, probs, s1


class MarginLoss(nn.Module):
    """
    Implementation of Margin Loss as described in paragraph 3 of the paper.
    In the original paper, the totall loss is the sum of all the individual losses
    for each digit capsule.

    Parameters
    -----------
    m_pos : decimal
    m_neg : decimal
    lambda_ : decimal
    """
    def __init__(self, m_pos, m_neg, lambda_):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, lengths, targets, size_average=True):
        """
        Performs the forward pass for the loss. The backward pass can be infereed automaticlly.
        Parameters
        ----------
        lengths : Tensor 
                    The norms of the vectors of each digit capsule on the last layer. 
                    Think of them as the predictions.
        targets : Tensor
        size_average : boolean
                    Determins weather the totall los will be computed by the sum (True) or the average
                    (False) of the losses for each digit. Default is True althow in the paper
                    the sum operation is performed.
        """
        t = torch.zeros(lengths.size()).long() # .long() is eqivalent to .to.(torch.int64)
        if targets.is_cuda:
            t = t.cuda()
        
        # Convert target to one-hot vectors.
        t = t.scatter_(1, targets.data.view(-1, 1), 1)

        targets = Variable(t)
        losses = targets.float() * F.relu(self.m_pos - lengths).pow(2) + \
            self.lambda_ * (1. - targets.float()) * F.relu(lengths - self.m_neg).pow(2)
        return losses.mean() if size_average else losses.sum()

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

def produce_splitters(train_size, valid_size=0.1):
    '''
    example of valid_size input: 0.2
    '''
    indices = list(range(train_size))
    split = int(np.floor(valid_size * train_size))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    return train_sampler, valid_sampler


if __name__ == '__main__':

    import argparse
    import torch.optim as optim
    from torchvision import datasets, transforms
    # from torch.autograd import Variable

    # Training settings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=str, default="MNIST",
                        help='choose between MNIST (default) or Fashion-MNIST or CIFAR10 or smallNORB')
    parser.add_argument('--dataset_file', type=str, default="../../data",
                        help='set the directory where the dataset is/will be located (default: ../../data)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 250)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status (default: 200)')
    parser.add_argument('--routing_iterations', help='# of routing iterations (default: 3).', type=int, default=3)
    parser.add_argument('--with_argmax' , action='store_true', default=False, help='If set then instead of calculating the parent capsule\'s vector by the weighted sum, we do argmax. (default: False)')
    parser.add_argument('--without_argmax_one' , action='store_false', default=True, help='Only used when with_argmax. If set (i.e. false), the winning child capsule (i) dose not pass its vote with coefficient c_i = 1 but instead \
    the vote is multiplied by the c_i which of course is the largest of all c_i for that parent capsule j.')
    parser.add_argument('--with_reconstruction', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    folder_name = os.path.join("./experiments",f"dynamic_routing_{date.today().strftime('%d-%m-%y')}_dataset:{args.dataset}_bsz:{args.batch_size}_epochs:{args.epochs}_lr:{args.lr}_routiter:{args.routing_iterations}_recon:{args.with_reconstruction}_argmax:{args.with_argmax}_withones:{args.without_argmax_one}")
    log = Logger(f"logfile.logs", folder_name)
    log.info_message("Parameters of the training procedure. \n")
    log.print_train_args(args)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        log.info_message("Using CUDA.")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # DataLoader is an iterator over the dataset which provides features like
    # data batching, shuffling and loading data in parallel using many workers.
    # The only transformations on images is the shift in any direction of up to 2 pixels.

    if args.dataset == "Fashion-MNIST":
        dataset_train = datasets.FashionMNIST(args.dataset_file, train=True, download=True,
                        transform=transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.2860,), (0.3530,)), # mean and std of FashionMNIST dataset
                            transforms.Pad(2), transforms.RandomCrop(28)
                        ]))
        train_sampler, valid_sampler = produce_splitters(len(dataset_train), 0.1)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, sampler=train_sampler, **kwargs)
        valid_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, sampler=valid_sampler, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(args.dataset_file, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
        output_classes = 10
        in_channels = 1
        types_of_primary_caps = 32
        input_image_dimension = 28

    elif args.dataset == "MNIST":
        dataset_train = datasets.FashionMNIST(args.dataset_file, train=True, download=True,
                        transform=transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)), # mean and std of MNIST dataset
                            transforms.Pad(2), transforms.RandomCrop(28)
                        ]))
        train_sampler, valid_sampler = produce_splitters(len(dataset_train), 0.1)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, sampler=train_sampler, **kwargs)
        valid_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, sampler=valid_sampler, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(args.dataset_file, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
        output_classes = 10
        in_channels = 1
        types_of_primary_caps = 32
        input_image_dimension = 28

    elif args.dataset == "CIFAR10":
        
        dataset_train = datasets.FashionMNIST(args.dataset_file, train=True, download=True,
                        transform=transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4821, 0.4466), (0.2470, 0.2435, 0.2616)), # mean and std of MNIST dataset
                            transforms.RandomCrop(28)
                        ]))
        train_sampler, valid_sampler = produce_splitters(len(dataset_train), 0.1)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, sampler=train_sampler, **kwargs)
        valid_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, sampler=valid_sampler, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(args.dataset_file, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(28),
                transforms.Normalize((0.4914, 0.4821, 0.4466), (0.2470, 0.2435, 0.2616))
            ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
        output_classes = 11 # 10 + "none-of-the-above" category (see Hinton's paper on dynamic routing section 7)
        in_channels = 3
        types_of_primary_caps = 64
        input_image_dimension = 28
    
    elif args.dataset == "smallNORB":
        transforms_train =transforms.Compose([
                transforms.Resize(48),
                transforms.RandomCrop(32),
                transforms.ColorJitter(brightness=32./255, contrast=0.5),
                transforms.ToTensor(),
            ])
        transforms_val =transforms.Compose([
                transforms.Resize(48),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
            ])
        transforms_test =transforms.Compose([
                transforms.Resize(48),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
            ])
        dataset_train = SmallNORB(args.dataset_file, train=True, download=True, transform=transforms_train)
        dataset_val = SmallNORB(args.dataset_file, train=True, download=True, transform=transforms_val)
        dataset_test = SmallNORB(args.dataset_file, train=False, download=True, transform=transforms_test)

        train_sampler, valid_sampler = produce_splitters(len(dataset_train), 0.1)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, sampler=train_sampler, **kwargs)
        valid_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, sampler=valid_sampler, **kwargs)

        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False, **kwargs)
        output_classes = 5
        in_channels = 2
        types_of_primary_caps = 64
        input_image_dimension = 32

    else:
        raise ValueError("Invalid dataset argument.")


    model = CapsNet(args.routing_iterations, n_classes=output_classes,
                    input_channels=in_channels, types_of_primary_caps=types_of_primary_caps, argmax=args.with_argmax, arg_max_ones=args.without_argmax_one)

    if args.with_reconstruction:
        reconstruction_model = ReconstructionNet(16, output_classes, in_channels, out_hw=input_image_dimension)
        reconstruction_alpha = 0.0005
        model = CapsNetWithReconstruction(model, reconstruction_model)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)

    # Set the margin loss as described in eq.4 of the paper. m+ = 0.9, m- = 0.1, lambda = 0.5.
    loss_fn = MarginLoss(0.9, 0.1, 0.5)

    def train(epoch):
        """
        Function used to train the Capsule network.
        """
        # Sets the mode flag to True. This is usefull when dropout and bachnorm because their
        # behavior differs from training to testing mode.
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()  # data is [batch_size, channels, hight_of_image, width_of_image]
            data, target = Variable(data), Variable(target, requires_grad=False) # Variables are depricated.
                                                                                # Use tensors with requires_grad instead.
            optimizer.zero_grad()
            if args.with_reconstruction:
                output, probs, _ = model(data, target)
                reconstruction_loss = F.mse_loss(output, data.view(-1, in_channels* 28 * 28)) # 28 == hight == width
                margin_loss = loss_fn(probs, target)
                loss = reconstruction_alpha * reconstruction_loss + margin_loss # Loss described in paragraph 4.1
            else:
                output, probs, _ = model(data)
                loss = loss_fn(probs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
            if batch_idx % args.log_interval == 0:
                log.info_message('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item()))

        train_loss /= len(train_loader.dataset)
        return train_loss
            

    def test(epoch, val_or_test_loader):
        """
        Function used to test the Capsule network.

        Returns
        ---------
        test_loss : scalar float number of the loss as computed by test set.
        """
        model.eval() # Change to no training mode.
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_or_test_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)# Variables are depricated.
                                                                                # Use tensors with requires_grad instead.
                                                                                # Volatile sets requires_grad=False and
                                                                                # is used when we do not use .backward()
                if args.with_reconstruction:
                    output, probs, _ = model(data, target)
                    reconstruction_loss = F.mse_loss(output, data.view(-1, in_channels* 28 * 28), reduction='sum').data.item()  # 28 == hight == width
                    test_loss += loss_fn(probs, target, size_average=False).data.item()
                    test_loss += reconstruction_alpha * reconstruction_loss
                else:
                    output, probs, _ = model(data)
                    test_loss += loss_fn(probs, target, size_average=False).data.item()

                pred = probs.data.max(1, keepdim=True)[1]  # get the index of the max probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_loss /= len(val_or_test_loader.dataset)
            acc = 100. * correct / len(val_or_test_loader.dataset)
            log.info_message('Test Epoch:{} Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(epoch, 
            test_loss, correct, len(val_or_test_loader.dataset), acc.data.item()))
        return test_loss, acc.data.item()

    log.info_message(f"Starting training... for {args.epochs} epochs.\n")
    test_loss =[]; train_loss = []; test_acc = []; best_loss1 = +Inf; best_acc1 = -1
    for epoch in range(1, args.epochs + 1):
        train_loss1 = train(epoch)
        test_loss1, test_acc1 = test(epoch, valid_loader)
        scheduler.step(test_loss1) # scheduler adapts the learning rate according to the evaluation loss 
                                    # (if it dosent decrease for more than patience steps, the learning rate is decreased).
        if test_loss1 < best_loss1 and test_acc1 > best_acc1:
            best_acc1 = test_acc1; best_loss1 = test_loss1
            torch.save(model.state_dict(),os.path.join(folder_name, '{:03d}_model_dict_{}routing_reconstruction{}.pth'.format(epoch, args.routing_iterations,
                                                                             args.with_reconstruction)))
                                                                    
        test_loss.append(test_loss1); train_loss.append(train_loss1); test_acc.append(test_acc1)
    log.info_message("Training finished.\n")

    # Make figures.
    log.info_message("Creating figures and saving lists of training data.")
    print_learning_curve(train_loss, test_loss, args.epochs).savefig(f"{os.path.join(folder_name, 'train_curve.png')}")
    print_accuracy_curve(test_acc, args.epochs).savefig(f"{os.path.join(folder_name, 'test_acc_curve.png')}")
    print_accuracy_curve_norm(test_acc, args.epochs).savefig(f"{os.path.join(folder_name, 'test_acc_curve_norm.png')}")

    # Save data to csv.
    train_loss_dict = {'Epoch' : np.arange(1,args.epochs + 1), 'train_loss' : train_loss}
    test_loss_dict = {'Epoch' : np.arange(1,args.epochs + 1), 'val_loss' : test_loss}
    test_acc_dict = {'Epoch' : np.arange(1,args.epochs + 1), 'val_acc' : test_acc}

    df_train_loss = pd.DataFrame(train_loss_dict) 
    df_test_loss = pd.DataFrame(test_loss_dict) 
    df_test_acc = pd.DataFrame(test_acc_dict) 

    df_train_loss.to_csv(f"{os.path.join(folder_name, 'train_loss.csv')}") 
    df_test_loss.to_csv(f"{os.path.join(folder_name, 'val_loss.csv')}") 
    df_test_acc.to_csv(f"{os.path.join(folder_name, 'val_acc.csv')}")
    log.info_message("Evaluating test loss and accuracy...")
    test_loss_all, test_acc_all = test(epoch, test_loader)
    log.info_message(f"Test loss: {test_loss_all},  Test accuracy: {test_acc_all}")
    log.info_message("Evaluation finished.")
    log.info_message("Finished.")
