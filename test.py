from base import *
from dataloader import DataLoader
from nn import Module, LinearLayer, MSELoss, CELoss
from optimizer import *



# Test code for the implemented classes and functions
# case 1: learn a sine function

# X = Tensor(np.linspace(-2*np.pi, 2*np.pi, 100).reshape(-1, 1), requires_grad=False)
# y = Tensor(np.sin(X.data), requires_grad=False)

# class Sinenet(Module):
#     def __init__(self):
#         super().__init__()
#         self.linear1 = LinearLayer(1, 10)
#         self.linear2 = LinearLayer(10, 10)
#         self.linear3 = LinearLayer(10, 1)
#         self.parameters = self.linear1.parameters + self.linear2.parameters
    
#     def forward(self, x):
#         x = self.linear1(x)
#         x = tanh(x)
#         x = self.linear2(x)
#         x = tanh(x)
#         x = self.linear3(x)
#         return x

#     def loss(self, pred, target):
#         return MSELoss(pred, target)
    
# model = Sinenet()
# optimizer = SGD(model.parameters)

# for epoch in range(100000):
#     pred = model(X)
#     loss = model.loss(pred, y)
#     loss.autograd()
#     optimizer.step()
#     optimizer.zero_grad()
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.data}")

# # print the learned function
# import matplotlib.pyplot as plt
# pred = model(X).data
# print(X.data.shape, pred.shape)

# plt.scatter(X.data.reshape(-1), y.data, label='True')
# plt.scatter(X.data.reshape(-1), pred, label='Predicted')
# plt.legend()
# plt.savefig("sine_fit.png")


# task 2: learn a classification boundary

X = Tensor(np.random.randn(200, 2), requires_grad=False)
y = Tensor((X.data[:, 0]**2 + X.data[:, 1]**2 < 1).reshape(-1, 1), dtype=np.float64, requires_grad=False)

class Circle(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = LinearLayer(2, 10)
        self.linear2 = LinearLayer(10, 10)
        self.linear3 = LinearLayer(10, 1)
        self.parameters = self.linear1.parameters + self.linear2.parameters
    
    def forward(self, x):
        x = self.linear1(x)
        x = tanh(x)
        x = self.linear2(x)
        x = tanh(x)
        x = self.linear3(x)
        return x
    def loss(self, pred, target):
        return CELoss(pred, target)
