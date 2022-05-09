import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd
from typing import Optional
from sklearn.base import TransformerMixin, BaseEstimator


class PatientDataset(torch.utils.data.Dataset):
    def __init__(self, X_df: pd.DataFrame, y: pd.Series, window: int):
        if X_df.shape[0] != len(y):
            raise ValueError("Received dataframe has different length than received label")
        self.X_df = X_df
        self.patients = X_df.id.unique()
        self.y = y
        self.window = window

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient = self.patients[idx]
        data = self.X_df[self.X_df['id'] == patient].drop('id', axis=1)
        labels = self.y[data.index]
        return torch.tensor(data.tail(self.window).values, dtype=torch.float32), \
               torch.tensor(labels.tail(self.window).max(), dtype=torch.float32)


# collate function to pad a batch
def batch_collate(batch):
    features_list, label_list = [], []

    for (_features, _label) in batch:
        features_list.append(_features)
        label_list.append(_label)

    features = pad_sequence(features_list, batch_first=True, padding_value=0)
    x_lengths = torch.LongTensor([len(x) for x in features_list])

    labels = torch.tensor(label_list, dtype=torch.float32)

    return (features, x_lengths), labels


class patientLSTM(nn.Module):
    def __init__(self, features_dim, hidden_dim, n_layers, out_dim=1, dropout=0):
        super(patientLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(features_dim, hidden_dim, num_layers=n_layers, dropout=dropout)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        padded_features, lengths = x
        packed_input = pack_padded_sequence(padded_features, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(packed_input)
        tag_space = self.hidden2tag(ht[-1])

        return tag_space
