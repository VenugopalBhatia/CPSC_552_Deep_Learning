from unicodedata import decimal
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from sklearn.datasets import fetch_california_housing

# import sample data
housing = fetch_california_housing(data_home="data")
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
X = torch.Tensor(housing["data"])
y = torch.Tensor(housing["target"]).unsqueeze(1)
#print(X.shape)
# create the weight vector
w_init = torch.randn(8, 1, requires_grad=True)
# TO DO:
# a) calculate closed form gradient with respect to the weights
X_cf = X.detach()
y_cf = y.detach()
w_cf = w_init.detach()
gradient_closed_form = 2*(np.dot(np.dot(X_cf.T,X_cf),w_cf)) - 2*(np.dot(X_cf.T,y_cf))
gradient_closed_form = torch.tensor(gradient_closed_form)
#print(gradient_closed_form)
# b) calculate gradient with respect to the weights w using autograd
# first create the loss function
loss = torch.square(torch.matmul(X,w_init) - y).sum()
#print("loss",loss)
loss.backward()
grad_pytorch = w_init.grad
#print(w_init)
# c) check that the two are equal
print(torch.allclose(grad_pytorch,gradient_closed_form))
print("closed form gradient",gradient_closed_form)
print("pytorch gradient",grad_pytorch)