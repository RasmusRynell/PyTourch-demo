import bz2
import pandas as pd
import spacy
import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nlp = spacy.load('en_core_web_lg')
with bz2.open('data/train.jsonl.bz2', 'rt') as source:
    df_train = pd.read_json(source, lines=True)

with bz2.open('data/dev.jsonl.bz2', 'rt') as source:
    df_dev = pd.read_json(source, lines=True)

labels = ['contradiction', 'entailment', 'neutral']

# Change from strings to integers
df_train['gold_label'] = df_train['gold_label'].apply(lambda x: labels.index(x))
df_dev['gold_label'] = df_dev['gold_label'].apply(lambda x: labels.index(x))


def preprocess(df):
    # Vectorize a single sentence.
    def transform_single_sentence(sentence):
        words = sentence.split(" ")
        result = np.zeros(nlp.vocab.vectors.shape[1])

        for word in words:
            result += nlp.vocab[word].vector
        
        return result

    # Vectorize a single row of the dataframe.
    def transform_row(row):
        sen1_vec = transform_single_sentence(row.sentence1)
        sen2_vec = transform_single_sentence(row.sentence2)
        return np.concatenate((sen1_vec, sen2_vec))

    # Vectorize the entire dataframe.
    def transform(df):
        return np.concatenate( [transform_row(row).reshape(1, -1) for row in df.itertuples()] )

    x = torch.from_numpy(transform(df)).float().to(device)
    y = torch.from_numpy(df['gold_label'].values).float().to(device)

    return (x, y)

xt, yt = preprocess(df_train)
xd, yd = preprocess(df_dev)


use_batch_norm = True
class Net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H, bias=not use_batch_norm)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(H)
        self.linear2 = torch.nn.Linear(H, H, bias=not use_batch_norm)
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(H)
        self.linear3 = torch.nn.Linear(H, H, bias=not use_batch_norm)
        self.relu3 = torch.nn.ReLU()
        self.bn3 = torch.nn.BatchNorm1d(H)
        self.linear4 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.bn1(x) if use_batch_norm else x
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.bn2(x) if use_batch_norm else x
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.bn3(x) if use_batch_norm else x
        x = self.linear4(x)
        return x

model = Net(xt.shape[1], 300, 3).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

iters = 100_000
batch_size = 32

all_losses_train = []
all_losses_dev = []

for curr_iter in range(iters+1):

    batch = np.random.choice(xt.shape[0], batch_size)
    x_batch = xt[batch]
    y_batch = yt[batch]

    y_pred = model(x_batch)

    loss = loss_fn(y_pred, y_batch.long())
    all_losses_train.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 10% of iterations.
    if curr_iter % (iters / 10) == 0:
        model.eval()
        print("Iteration: %d, Loss: %f" % (curr_iter, loss.item()))

        # Test the model
        with torch.no_grad():
            y_pred = model(xd)
            y_pred_v = torch.argmax(y_pred, dim=1)

            current_loss = loss_fn(y_pred, yd.long())
            all_losses_dev.append(current_loss.item())

            accuracy = (y_pred_v == yd).sum().item() / len(yd)
            print(f"Accuracy: {accuracy:.3f}")
        
        model.train()


plt.plot(all_losses_train, label='train')
plt.plot([i * (iters / 10) for i in range(0, 11)], all_losses_dev, label='dev')
plt.legend()
plt.show()