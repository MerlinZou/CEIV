'''
This is the code used for academic research in College English IV course 

Multivariate Linear Regression
Hong-Gang Zou, Yu-Tai Li, Li-Hang Hu, Chen-Hao Zhang
Group 3, Class 1, University of Chinese Academy of Sciences
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

x_values = [[44,2,4.06],[56,3,4.15],[56,3,4.15],[36,2,2.07],[20,2,4.47],[48,3,3.75],[64,2,4.57],[80,4,5.03],[24,3,1.87],[104,3,8.32],[60,4,9.6],[92,3,5.64],[30,3,1.94],[64,1,5.67],[70,1,5.55],[70,3,3.05],[56,3,4.26],[80,4,9.6],[74,4,9.6],[106,4,7.35],[42,3,4.55],[88,3,2.37],[64,2,6.11],[64,2,5.57],[24,1,2.47],[72,2,7.32],[74,4,9.6],[64,2,5.57],[64,2,5.57],[84,3,5.63],[64,2,6.85]]
x_train = np.array(x_values, dtype=np.float32)
print(x_train)
 
y_values = [0.815,0.818,0.876,0.830,0.887,0.867,0.895,1.017,0.823,3.770,1.169,2.038,0.875,0.891,0.921,0.906,1.251,1.195,1.187,1.419,0.904,0.779,1.303,0.972,0.771,1.078,1.128,1.314,1.898,1.733,0.970]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1,1)
print(y_train)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel,self).__init__()      
        self.linear = nn.Linear(input_dim, output_dim)   
        
    def forward(self, x):
        out = self.linear(x)
        return out
    
input_dim = 3
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)
criterion = nn.MSELoss()
learning_rate = 0.00005
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
 
epochs = 10000000
for epoch in range(epochs):
    epoch += 1
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train)) 
    optimizer.zero_grad()    
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    #print("while training: \n", model.state_dict())
    print(loss)
    
print("After training: \n", model.state_dict())
