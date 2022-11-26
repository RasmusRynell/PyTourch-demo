import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time


data = pd.read_csv("data/TempTullinge.csv", sep=";", header=0)
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%y')
data['date'] = data['date'].map(datetime.datetime.toordinal)
data['date'] = data['date'] - 733773

# Scale the data with standardization
data['date'] = (data['date'] - data['date'].mean()) / data['date'].std()
data['temp'] = (data['temp'] - data['temp'].mean()) / data['temp'].std()

# Divide the data into training and testing
train = data.sample(frac=0.8, random_state=200)
test = data.drop(train.index)


# # Convert the data to tensors date as input and temp as output
train_x = torch.tensor(train[['date']].values, dtype=torch.float32)
train_y = torch.tensor(train['temp'].values, dtype=torch.float32)
test_x = torch.tensor(test[['date']].values, dtype=torch.float32)
test_y = torch.tensor(test['temp'].values, dtype=torch.float32)

# # Create a neural network
class Net(torch.nn.Module):
    def __init__(self, n_hidden_neurons1, n_hidden_neurons2):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons1)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons1, n_hidden_neurons2)
        self.act2 = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(n_hidden_neurons2, n_hidden_neurons2)
        self.act3 = torch.nn.Sigmoid()
        self.fc4 = torch.nn.Linear(n_hidden_neurons2, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)
        return x
    

net = Net(50, 50)

# Create a loss function and an optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# Train the network
print('Training...', flush=True)
start = time.time()
epochs = 100_00
for epoch in range(1,epochs+1):
    optimizer.zero_grad()
    
    y_pred = net.forward(train_x.unsqueeze(1))
    loss = loss_fn(y_pred.squeeze(), train_y)
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}", flush=True)
end = time.time()
print(f"Training took {end-start:.2f} seconds", flush=True)

# Test the network
y_pred = net.forward(test_x.unsqueeze(1))
loss = loss_fn(y_pred.squeeze(), test_y)
print(f"Test loss: {loss.item():.4f}", flush=True)


# Plot the results
plt.scatter(test_x, test_y, label='Actual')
plt.scatter(test_x, y_pred.detach().numpy(), label='Predicted')
plt.legend()
plt.show()
