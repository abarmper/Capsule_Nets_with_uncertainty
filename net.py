"""CapsNet with MNIST

This code implements Capsule Network as it is presented in the paper "Dynamic Routing Between Capsules"
by Sasabour, Frosst and  Hinton. I will try to comment it step by step so as to better understand this implementation.
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.optim import lr_scheduler
from torch.autograd import Variable

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
    lengths2 = x.pow(2).sum(dim=2) # vector s_j is stored in the third dimension of x.
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

    Methods
    -------
    forward
    """
    def __init__(self, input_caps, output_caps, n_iterations):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        # Parameters aare like tensors but when assigned as Module attribiutes they are added to the parameters list.
        # Parameters go to cuda if model is on cuda.
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps))) 

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b, dim=1)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = F.softmax(b_batch.view(-1, output_caps), dim=1).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)

        return v


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
        v = self.routing_module(u_predict)
        return v


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
        out = out.view(N, self.output_caps, self.output_dim, H, W) # Group the output vectors together.

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(out.size(0), -1, out.size(4))
        out = squash(out)
        return out


class CapsNet(nn.Module):
    """
    Putting it all together. It encapsulates the network found in Figure 1 of the paper.
    """
    def __init__(self, routing_iterations, n_classes=10, input_channels=1, types_of_primary_caps=32):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=9, stride=1)
        self.primaryCaps = PrimaryCapsLayer(256, types_of_primary_caps, 8, kernel_size=9, stride=2)  # outputs 6*6
        self.num_primaryCaps = types_of_primary_caps * 6 * 6
        routing_module = AgreementRouting(self.num_primaryCaps, n_classes, routing_iterations)
        self.digitCaps = CapsLayer(self.num_primaryCaps, 8, n_classes, 16, routing_module)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.primaryCaps(x)
        x = self.digitCaps(x)
        probs = x.pow(2).sum(dim=2).sqrt()
        return x, probs


class ReconstructionNet(nn.Module):
    """
    Decoder network in figure 2. Acts as regularizer.
    """
    def __init__(self, n_dim=16, n_classes=10, in_channels=1):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim * n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, in_channels * 28 * 28) # 28 == hight == width
        self.n_dim = n_dim
        self.n_classes = n_classes

    def forward(self, x, target):
        # Mask so as to feed the decoder network with the DigitCaps vector prediction of the correct digit.
        # Dose the same apply during test time? (Where we do not have the target.)
        # Shouldn't we use the argmax() over the norms of digitCaps vectors during test time instead ?
        mask = Variable(torch.zeros((x.size()[0], self.n_classes)), requires_grad=False)

        if next(self.parameters()).is_cuda:
            mask = mask.cuda()
        if self.training:
            mask.scatter_(1, target.view(-1, 1), 1.) # To one hot vector
        else:
            mask.scatter_(1, torch.argmax(torch.argmax(torch.norm(x, dim=2), dim=1, keepdim=False).view(-1, 1), dim=1,keepdim=True), 1.) # To one hot vector
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
        x, probs = self.capsnet(x)
        reconstruction = self.reconstruction_net(x, target)
        return reconstruction, probs


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


if __name__ == '__main__':

    import argparse
    import torch.optim as optim
    from torchvision import datasets, transforms
    # from torch.autograd import Variable

    # Training settings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=str, default="MNIST",
                        help='choose between MNIST (default) or Fashion-MNIST or CIFAR10')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--routing_iterations', help='# of routing iterations (default: 3)', type=int, default=3)
    parser.add_argument('--with_reconstruction', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # DataLoader is an iterator over the dataset which provides features like
    # data batching, shuffling and loading data in parallel using many workers.
    # The only transformations on images is the shift in any direction of up to 2 pixels.
    if args.dataset == "Fashion-MNIST":
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Pad(2), transforms.RandomCrop(28),
                            transforms.ToTensor()
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor()
            ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
        output_classes = 10
        in_channels = 1
        types_of_primary_caps = 32
    elif args.dataset == "MNIST":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Pad(2), transforms.RandomCrop(28),
                            transforms.ToTensor()
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor()
            ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
        output_classes = 10
        in_channels = 1
        types_of_primary_caps = 32
    elif args.dataset == "CIFAR10":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Pad(2), transforms.RandomCrop(28),
                            transforms.ToTensor()
                        ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                transforms.CenterCrop(28), transforms.ToTensor()
            ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

        output_classes = 11 # 10 + "none-of-the-above" category (see Hinton's paper section 7)
        in_channels = 3
        types_of_primary_caps = 64
    else:
        raise ValueError("Invalid dataset argument.")


    model = CapsNet(args.routing_iterations, n_classes=output_classes,
                    input_channels=in_channels, types_of_primary_caps=types_of_primary_caps)

    if args.with_reconstruction:
        reconstruction_model = ReconstructionNet(16, output_classes, in_channels)
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
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()  # data is [batch_size, channels, hight_of_image, width_of_image]
            data, target = Variable(data), Variable(target, requires_grad=False) # Variables are depricated.
                                                                                # Use tensors with requires_grad instead.
            optimizer.zero_grad()
            if args.with_reconstruction:
                output, probs = model(data, target)
                reconstruction_loss = F.mse_loss(output, data.view(-1, in_channels* 28 * 28)) # 28 == hight == width
                margin_loss = loss_fn(probs, target)
                loss = reconstruction_alpha * reconstruction_loss + margin_loss # Loss described in paragraph 4.1
            else:
                output, probs = model(data)
                loss = loss_fn(probs, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item()))

    def test():
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
            for data, target in test_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)# Variables are depricated.
                                                                                # Use tensors with requires_grad instead.
                                                                                # Volatile sets requires_grad=False and
                                                                                # is used when we do not use .backward()
                if args.with_reconstruction:
                    output, probs = model(data, target)
                    reconstruction_loss = F.mse_loss(output, data.view(-1, in_channels* 28 * 28), reduction='sum').data.item()  # 28 == hight == width
                    test_loss += loss_fn(probs, target, size_average=False).data.item()
                    test_loss += reconstruction_alpha * reconstruction_loss
                else:
                    output, probs = model(data)
                    test_loss += loss_fn(probs, target, size_average=False).data.item()

                pred = probs.data.max(1, keepdim=True)[1]  # get the index of the max probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        return test_loss

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test_loss = test()
        scheduler.step(test_loss) # scheduler adapts the learning rate according to the evaluation loss 
                                    # (if it dosent decrease for more than patience steps, the learning rate is decreased).
        torch.save(model.state_dict(),
                   '{:03d}_model_dict_{}routing_reconstruction{}.pth'.format(epoch, args.routing_iterations,
                                                                             args.with_reconstruction))
