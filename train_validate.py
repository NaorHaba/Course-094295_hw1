import argparse
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random

from score_functions import F1
from LSTMTrainer import RNNTrainer
from torch.utils.data import DataLoader

from patient_LSTM import TrainDataset, patientLSTM, batch_collate
random.seed(42)


def split_features_label(df):

    return df.drop(['SepsisLabel'], axis=1), df['SepsisLabel'].astype(int)


def split_train_validation(valid_size):

    sick = set(t_df[t_df['SepsisLabel'] == 1.0]['id'].unique())
    healthy = set(t_df['id'].unique()) - sick
    t_sick = set(random.sample(sick, int(len(sick) * (1 - valid_size))))
    v_sick = sick - t_sick
    t_healthy = set(random.sample(healthy, int(len(healthy) * (1 - valid_size))))
    v_healthy = healthy - t_healthy

    train = t_df[t_df.id.isin(list(t_sick) + list(t_healthy))].sort_values(['id', 'SepsisLabel'])
    valid = t_df[t_df.id.isin(list(v_sick) + list(v_healthy))].sort_values(['id', 'SepsisLabel'])

    return train, valid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and Validate model')

    # Paths
    parser.add_argument('train_file', type=str, help='path to train csv file')
    parser.add_argument('test_file', type=str, help='path to test csv file')
    parser.add_argument('valid_size', type=float, help='size of validation data (proportion)')
    # Data parameters
    parser.add_argument('sampling_method', type=str, help='method for sampling')
    parser.add_argument('remove_columns', type=str, help='method for columns removal')
    parser.add_argument('scale_method', type=str, help='method for scaling')
    # Model parameters
    parser.add_argument('batch_size', type=str, help='batch size')
    parser.add_argument('train_batch_size', type=float, help='train batch size')
    parser.add_argument('test_batch_size', type=float, help='test batch size')
    parser.add_argument('window_size', type=int, help='window size')
    parser.add_argument('loss_fn', type=str, help='loss function for model')
    parser.add_argument('optimizer', type=str, help='optimizer for model')
    parser.add_argument('hidden_dim', type=int, help='')
    parser.add_argument('LSTM_n_layers', type=int, help='')
    parser.add_argument('dropout', type=float, help='')
    parser.add_argument('lr', type=float, help='')
    parser.add_argument('true_threshold', type=float, help='')
    parser.add_argument('epochs', type=int, help='')

    args = parser.parse_args()

    # read data
    print('Reading CSV...')
    t_df = pd.read_csv(args.train_file, index_col=0, dtype='float')
    test = pd.read_csv(args.test_file, index_col=0, dtype='float')
    print('Done')

    t_df['id'] = t_df['id'].astype(int)
    test['id'] = test['id'].astype(int)

    # split train-validation
    train, valid = split_train_validation(args.valid_size)

    sampler = None
    # Under/Over-Sampling
    if args.sampling_method is not None:
        # TODO change to different functions
        if args.sampling_method == 'sample_health_records':
            pass
        remove_amount = len(t_healthy) - 2 * len(t_sick)
        # remove_amount = 5000
        remove_healthy = random.sample(healthy, remove_amount)
        train = train[~train['id'].isin(remove_healthy)]

    # Remove columns
    if args.remove_columns is not None:
        # TODO change to different functions
        # train = train.drop(train.columns[8: 34], axis=1)
        # valid = valid.drop(valid.columns[8: 34], axis=1)
        pass

    # scale columns
    if args.scale_method is not None:
        # TODO change to different functions
        scaling_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'Age', 'HospAdmTime',
                           'ICULOS']  # rest are already scaled
        scaler = MinMaxScaler()
        train[scaling_columns] = scaler.fit_transform(train[scaling_columns])
        valid[scaling_columns] = scaler.transform(valid[scaling_columns])
        test[scaling_columns] = scaler.transform(test[scaling_columns])

    # create data loaders
    train_X, train_y = split_features_label(train)
    valid_X, valid_y = split_features_label(valid)
    test_X, test_y = split_features_label(test)

    train_ds = TrainDataset(train_X, train_y, args.window_size)
    valid_ds = TrainDataset(valid_X, valid_y, args.window_size)
    test_ds = TrainDataset(test_X, test_y, args.window_size)

    if sampler is not None:
        train_dl = DataLoader(train_ds, batch_size=args.train_batch_size, collate_fn=batch_collate, sampler=sampler)
    else:
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=batch_collate, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=args.test_batch_size, collate_fn=batch_collate)
    test_dl = DataLoader(test_ds, batch_size=args.test_batch_size, collate_fn=batch_collate)

    # build model
    model = patientLSTM(features_dim=train_X.shape[1], hidden_dim=args.hidden_dim, n_layers=args.LSTM_n_layers,
                        dropout=args.dropout)

    loss_fn = nn.BCEWithLogitsLoss()
    score_fn = F1()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    trainer = RNNTrainer(model, loss_fn, optimizer, true_threshold=args.true_threshold)

    # train and test
    trainer.fit(train_dl, valid_dl, args.epochs, score_fn=score_fn, checkpoints=f'models/LSTM_window{args.window_size}',
                early_stopping=15)

    trainer.test(test_dl, score_fn=score_fn)
