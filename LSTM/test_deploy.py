import pickle

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from LSTM.LSTMTrainer import RNNTrainer
from LSTM.patient_LSTM import PatientDataset, batch_collate, patientLSTM
from LSTM.train_validate import under_sample, weighted_over_sampler, scaler_fn, split_features_label
import random

from utils.score_functions import F1

random.seed(42)

# train deploy and test using the best parameters we found
if __name__ == '__main__':
    print('Reading CSV...')
    train = pd.read_csv('../data/train_raw.csv', index_col=0, dtype='float')
    test = pd.read_csv('../data/test_raw.csv', index_col=0, dtype='float')
    print('Done')

    train['id'] = train['id'].astype(int)
    test['id'] = test['id'].astype(int)

    # Under-Over-Sampling
    train = under_sample(train, under_sample_rate=0.5)
    sampler = weighted_over_sampler(train)

    # # Remove columns
    # if args.remove_columns is not None:
    #     # TODO change to different functions
    #     # train = train.drop(train.columns[8: 34], axis=1)
    #     # valid = valid.drop(valid.columns[8: 34], axis=1)
    #     pass

    # scale columns
    scaler = StandardScaler()
    scaling_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'Age', 'HospAdmTime', 'ICULOS']
    train = scaler_fn(scaler, scaling_columns, train)
    test = scaler_fn(scaler, scaling_columns, test, test=True)
    pickle.dump(scaler, open('../models/LSTM_scaler.pkl', 'wb'))

    # create data loaders
    train_X, train_y = split_features_label(train)
    test_X, test_y = split_features_label(test)

    train_ds = PatientDataset(train_X, train_y, 35)
    test_ds = PatientDataset(test_X, test_y, 35)

    train_dl = DataLoader(train_ds, batch_size=32, collate_fn=batch_collate, sampler=sampler)
    test_dl = DataLoader(test_ds, batch_size=32, collate_fn=batch_collate)

    # build model
    model = patientLSTM(features_dim=train_X.shape[1] - 1, hidden_dim=3, n_layers=512,
                        dropout=0.45)

    loss_fn = nn.BCEWithLogitsLoss()
    score_fn = F1()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, betas=(0.9, 0.999))

    trainer = RNNTrainer(model, loss_fn, optimizer, true_threshold=0.5)

    model_name = f'../models/LSTM_final_model.pth'
    trainer.fit(train_dl, test_dl, 70, score_fn=score_fn, checkpoints=model_name, early_stopping=15)

    trainer.model.load_state_dict(torch.load(model_name)['model_state'])

    trainer.test(test_dl, score_fn=score_fn)
