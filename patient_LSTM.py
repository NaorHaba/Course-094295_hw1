import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd
from typing import Optional
from sklearn.base import TransformerMixin, BaseEstimator


class PatientDataset(torch.utils.data.Dataset):
    def __init__(self, X_df: pd.DataFrame, y: pd.Series, scaler: Optional[TransformerMixin, BaseEstimator] = None):
        if X_df.shape[0] != len(y):
            raise ValueError("Received dataframe has different length than received label")
        self.X_df = X_df
        self.patients = X_df.id.unique()
        self.y = y
        self.scaler = scaler

    def __len__(self):
        return self.X_df

    def __getitem__(self, idx):
        patient = self.patients[idx]
        data = self.X_df[self.X_df['id'] == patient]
        if self.scaler is not None:
            data = self.scaler.transform(data)
        labels = self.y[data.index]
        return data, labels


class TrainDataset(PatientDataset):
    def __init__(self, X_df: pd.DataFrame, y: pd.Series, window: int, scaler: Optional[TransformerMixin] = None):
        super().__init__(X_df, y, scaler)
        self.window = window

    def __len__(self):
        return self.window

    def __getitem__(self, idx):
        data, labels = super().__getitem__(idx)
        return torch.tensor(data.tail(self.window)), torch.tensor(labels.tail(self.window).max(), dtype=torch.int64)


class TestDataset(PatientDataset):
    def __init__(self, X_df: pd.DataFrame, y: pd.Series, window: int, scaler: Optional[TransformerMixin] = None):
        super().__init__(X_df, y, scaler)
        self.window = window

    def __getitem__(self, idx):
        data, labels = super().__getitem__(idx)
        data_iter = data.rolling(self.window)
        return (torch.tensor(df) for df in data_iter), torch(labels.max(), dtype=torch.int64)


# collate function to pad a batch
def batch_collate(batch):
    label_list, features_list, = [], []

    for (_features, _label) in batch:
        label_list.append(_label)
        features_list.append(_features)

    labels = torch.tensor(label_list, dtype=torch.int64)

    features = pad_sequence(features_list, batch_first=True, padding_value=0)
    x_lengths = torch.LongTensor([len(x) for x in features_list])

    return (features, x_lengths), labels


class patientLSTM(nn.Module):
    def __init__(self, features_dim, hidden_dim, n_layers, out_dim=1, dropout=0):
        super(patientLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(features_dim, hidden_dim, num_layers=n_layers, dropout=dropout)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, out_dim)

    def forward(self, padded_features, lengths):
        packed_input = pack_padded_sequence(padded_features, lengths, batch_first=True)
        packed_output, (ht, ct) = self.lstm(packed_input)
        tag_space = self.hidden2tag(ht[-1])

        return tag_space