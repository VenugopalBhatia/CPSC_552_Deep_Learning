""" Neural Network.
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with PyTorch. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
"""

import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # activation functions
import torch.optim as optim  # optimizer
from torch.autograd import Variable # add gradients to tensors
from torch.nn import Parameter # model parameter functionality
from tqdm import trange
import torchvision
import torchvision.datasets as datasets
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ##################################
# import data
# ##################################
# download the MNIST dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

# separate into data and labels
# training data

train_data = mnist_trainset.data.to(dtype=torch.float32)
train_data = train_data.reshape(-1, 784)
train_labels = mnist_trainset.targets.to(dtype=torch.long)

print("train data shape: {}".format(train_data.size()))
print("train label shape: {}".format(train_labels.size()))

# testing data
test_data = mnist_testset.data.to(dtype=torch.float32)[:2000]
test_data = test_data.reshape(-1, 784)
test_labels = mnist_testset.targets.to(dtype=torch.long)[:2000]

print("test data shape: {}".format(test_data.size()))
print("test label shape: {}".format(test_labels.size()))

# load into torch datasets
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

# ##################################
# set hyperparameters
# ##################################
# Parameters
learning_rate = 0.05 # Ha ha! This means it will learn really quickly, right?
num_epochs = 100 
batch_size = 128

# Network Parameters
n_hidden_1 = 75  # 1st layer number of neurons
n_hidden_2 = 64
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)

# ##################################
# defining the model 
# ##################################

# method 1: define a python class, which inherits the rudimentary functionality of a neural network from nn.Module

class Fully_Connected_Neural_Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Fully_Connected_Neural_Net, self).__init__()
        
        # As you'd guess, these variables are used to set the number of dimensions coming in and out of the network. We supply them when we initialize the neural network class.
        # Adding them to the class as variables isn't strictly necessary in this case -- but it's good practice to do this book-keeping, should you need to reference the input dim from somewhere else in the class.
        self.input_dim = input_dim
        self.output_dim = output_dim

        # And here we have the PyTorch magic -- a single call of nn.Linear creates a single fully connected layer with the specified input and output dimensions. All of the parameters are created automatically.
        self.layer1 = nn.Linear(input_dim, n_hidden_1)
        #self.layer2 = nn.Linear(n_hidden_1, output_dim)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, output_dim)
        
        # TODO: Create possible extra layers here

        # You can find many other nonlinearities on the PyTorch docs.
        self.nonlin1 = nn.ReLU()
        # TODO: You might try some different activation functions here
        self.nonlin2 = nn.ReLU()
        self.nonlin3 = nn.Sigmoid()
    def forward(self, x):
        # When you give your model data (e.g., by running `model(data)`, the data gets passed to this forward function -- the network's forward pass.
        
        # You can very easily pass this data into your model's layers, reassign the output to the variable x, and continue.
        # TODO: Play with the position of the nonlinearity, and with the number of layers
        x = self.layer1(x)
        x = self.nonlin1(x)
        x = self.layer2(x)
        x = self.nonlin2(x)
        x = self.layer3(x)
        x = self.nonlin3(x)
        

        return x

# alternative way of defining a model in pytorch
# you can create an equivalent model to FCNN above
# using nn.Sequential
#
# model2 = nn.Sequential(nn.Linear(num_input, n_hidden_1),
#                        nn.Sigmoid(),
#                        nn.Linear(n_hidden_1, n_hidden_2),
#                        nn.Sigmoid(),
#                        nn.Linear(n_hidden_2, num_classes))

# ##################################
# helper functions
# ##################################
def get_accuracy(output, targets):
    """calculates accuracy from model output and targets
    """
    output = output.detach()
    predicted = output.argmax(-1)
    correct = (predicted == targets).sum().item()

    accuracy = correct / output.size(0) * 100

    return accuracy


def to_one_hot(y, c_dims=10):
    """converts a N-dimensional input to a NxC dimnensional one-hot encoding
    """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    c_dims = c_dims if c_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], c_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

def lp_reg(params,p=1):
    sum = 0
    for w in params:
        if len(w.shape) > 1: # if this isn't a bias
            sum += torch.sum(w**p)
    return sum ** (1/p)

# ##################################
# creata dataloader
# ##################################
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)

# These might be handy
CELoss = torch.nn.CrossEntropyLoss()
softmax = torch.nn.Softmax()
sigmoid = torch.nn.Sigmoid()

# ##################################
# main training function
# ##################################

def train():

    model = Fully_Connected_Neural_Net(num_input, num_classes)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # initialize loss list
    metrics = [[0, 0]]

    # iterate over epochs
    for ep in trange(num_epochs):
        model.train()

        # iterate over batches
        for batch_indx, batch in enumerate(trainloader):

            # unpack batch
            data, labels = batch

            ###################
            # TO DO
            ##################
            # Complete the training loop by feeding the data to the model, comparing the output to the actual labels to compute a loss, backpropogating the loss, and updating the model parameters.
            
            # This is the code that runs every batch
            # ...
        # And here you might put things that run every epoch
        # ...
            optimizer.zero_grad()
            preds = model(data)
            loss = CELoss(preds,labels)
            loss.backward()
            optimizer.step()

        # compute full train and test accuracies 
        # every epoch
        model.eval()# model will not calculate gradients for this pass, and will disable dropout
        train_ep_pred = model(train_data)
        test_ep_pred = model(test_data)

        train_accuracy = get_accuracy(train_ep_pred, train_labels)
        test_accuracy = get_accuracy(test_ep_pred, test_labels)

        # print loss every 100 epochs
        if ep % 10 == 0:
            print("train acc: {}\t test acc: {}\t at epoch: {}".format(train_accuracy,test_accuracy,ep))
        metrics.append([train_accuracy, test_accuracy])
    
    test_preds_cm = model(test_data)
    test_preds_cm = test_preds_cm.argmax(-1)

    print(confusion_matrix(test_labels,test_preds_cm))


    return np.array(metrics), model

# so using the training function, you would ultimately 
# be left with your metrics (in this case accuracy vs epoch) and 
# your trained model.
# Ex. 
#metric_array, trained_model = train()


def plot_errors_v_epoch(metric_array,title):
    epochs = np.arange(len(metric_array))
    plt.plot(epochs,100-metric_array[:,0], label="train")
    plt.plot(epochs,100-metric_array[:,1], label="test")
    plt.legend()
    plt.title(title)
    plt.show()


    

def plot_accuracies_v_epoch(metric_array,title):
    epochs = np.arange(len(metric_array))
    plt.plot(epochs,metric_array[:,0], label="train")
    plt.plot(epochs,metric_array[:,1], label="test")
    plt.legend()
    plt.title(title)
    plt.show()
    

if __name__ == '__main__':
    metrics, model = train()
    plot_accuracies_v_epoch(metrics,"Training and Testing Accuracies")
    #plot_errors_v_epoch(metrics,"Training and Test Errors")
