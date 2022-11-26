import 1§torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# create a pandas dataframe
data = pd.DataFrame({'x': [], 'y': []})

width = 35
pints_per_width = 50
data['x'] = pd.Series([np.random.uniform(0, width) for i in range(width*pints_per_width)])
data['is_cos'] = pd.Series([np.random.randint(0, 2) for i in range(width*pints_per_width)])
data['y'] = pd.Series([np.cos(x) if is_cos else np.sin(x) for x, is_cos in zip(data['x'], data['is_cos'])])

print(data.shape)

# Divide the data into training and testing
train = data.sample(frac=0.8, random_state=200)
test = data.drop(train.index)

# Convert the data to tensors x and is_cos as input and y as output
train_x = torch.tensor(train[['x', 'is_cos']].values, dtype=torch.float32)
train_y = torch.tensor(train['y'].values, dtype=torch.float32)
test_x = torch.tensor(test[['x', 'is_cos']].values, dtype=torch.float32)
test_y = torch.tensor(test['y'].values, dtype=torch.float32)

# Create a neural network
class Net(torch.nn.Module):
    def __init__(self, n_hidden_neurons1, n_hidden_neurons2):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(2, n_hidden_neurons1)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons1, n_hidden_neurons2)
        self.act2 = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(n_hidden_neurons2, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x
    

net = Net(25, 25)

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
    
    if epoch%(epochs*0.1) == 0:
        print(f'Epoch: {epoch}/{epochs} ({epoch/epochs if epoch != 0 else 0:.1%}) Loss: {loss.item():.4f}', flush=True)

end = time.time()
print(f'Training took {end-start:.2f} seconds', flush=True)

y_pred = net.forward(test_x.unsqueeze(1)).detach().numpy()

# Plot the results, sin as blue and cos as red
plt.scatter(test_x[:,0][test_x[:,1] == 0], test_y[test_x[:,1] == 0], c='b', label='sin')
plt.scatter(test_x[:,0][test_x[:,1] == 1], test_y[test_x[:,1] == 1], c='r', label='cos')
plt.scatter(test_x[:,0][test_x[:,1] == 0], y_pred[test_x[:,1] == 0], c='b', marker='x', label='sin pred')
plt.scatter(test_x[:,0][test_x[:,1] == 1], y_pred[test_x[:,1] == 1], c='r', marker='x', label='cos pred')
plt.legend()
plt.show()