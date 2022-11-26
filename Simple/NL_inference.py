import bz2
import pandas as pd


with bz2.open('train.jsonl.bz2', 'rt') as source:
    df_train = pd.read_json(source, lines=True, nrows=25000)
    print('Number of sentence pairs in the training data:', len(df_train))

with bz2.open('train.jsonl.bz2', 'rt') as source:
    df_train_full = pd.read_json(source, lines=True)

with bz2.open('dev.jsonl.bz2', 'rt') as source:
    df_dev = pd.read_json(source, lines=True)
    print('Number of sentence pairs in the development data:', len(df_dev))