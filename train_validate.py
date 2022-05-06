import argparse
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import wandb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random

from score_functions import F1
from LSTMTrainer import RNNTrainer
from torch.utils.data import DataLoader, WeightedRandomSampler

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


def under_sample(train, under_sample_rate):
    healthy = set(train[train['SepsisLabel'] == 0.0]['id'].unique())
    remove_healthy = random.sample(healthy, (1 - under_sample_rate) * len(healthy))
    train = train[~train['id'].isin(remove_healthy)]
    return train


def weighted_over_sampler(train):
    samples_weight = []
    healthy = set(train[train['SepsisLabel'] == 0.0]['id'].unique())
    sick = set(train[train['SepsisLabel'] == 1.0]['id'].unique())
    for patient, label in train.groupby('id')['SepsisLabel'].max():
        if label == 1.0:
            samples_weight.append(1. / len(sick))
        else:
            samples_weight.append(1. / len(healthy))
    samples_weight = torch.tensor(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    return sampler


def scaler_fn(scaler, scaling_columns, train, valid, test):
    train[scaling_columns] = scaler.fit_transform(train[scaling_columns])
    valid[scaling_columns] = scaler.transform(valid[scaling_columns])
    test[scaling_columns] = scaler.transform(test[scaling_columns])
    return train, valid, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and Validate model')

    # Paths
    parser.add_argument('train_file', type=str, help='path to train csv file',
                        default='data/train_raw.csv')
    parser.add_argument('test_file', type=str, help='path to test csv file',
                        default='data/test_raw.csv')
    parser.add_argument('valid_size', type=float, help='size of validation data (proportion)',
                        default=0.15)
    # Data parameters
    parser.add_argument('sampling_method', type=str, help='method for sampling',
                        default='under')
    parser.add_argument('under_sample_rate', type=float, help='sampling rate for under-sampling',
                        default=0.5)
    parser.add_argument('remove_columns', type=str, help='method for columns removal',
                        default='')
    parser.add_argument('scale_method', type=str, help='method for scaling',
                        default='standard')
    parser.add_argument('scaling_columns', type=str, help='method for columns removal',
                        default='')  # '_' delimited: "HR_Temp_..."
    # Model parameters
    parser.add_argument('batch_size', type=float, help='batch size',
                        default=30)
    parser.add_argument('train_batch_size', type=float, help='train batch size',
                        default=30)
    parser.add_argument('test_batch_size', type=float, help='test batch size',
                        default=30)
    parser.add_argument('window_size', type=int, help='window size',
                        default=30)
    parser.add_argument('hidden_dim', type=int, help='',
                        default=600)
    parser.add_argument('LSTM_n_layers', type=int, help='',
                        default=3)
    parser.add_argument('dropout', type=float, help='',
                        default=0.4)
    parser.add_argument('lr', type=float, help='',
                        default=0.01)
    parser.add_argument('true_threshold', type=float, help='',
                        default=0.5)
    parser.add_argument('epochs', type=int, help='',
                        default=100)

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
        if 'under' in args.sampling_method:
            train = under_sample(train, under_sample_rate=args.under_sample_rate)
        elif 'over' in args.sampling_method:
            sampler = weighted_over_sampler(train)
        else:
            train = under_sample(train, under_sample_rate=args.under_sample_rate)
            sampler = weighted_over_sampler(train)

    # Remove columns
    if args.remove_columns is not None:
        # TODO change to different functions
        # train = train.drop(train.columns[8: 34], axis=1)
        # valid = valid.drop(valid.columns[8: 34], axis=1)
        pass

    # scale columns
    if args.scale_method is not None:
        if args.scale_method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        if not args.scaling_columns:
            scaling_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'Age', 'HospAdmTime',
                               'ICULOS']  # rest are already scaled
        else:
            scaling_columns = args.scaling_columns.split('_')
        train, valid, test = scaler_fn(scaler, scaling_columns, train, valid, test)

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
    model_name = f'{wandb.run.dir}/model'
    trainer.fit(train_dl, valid_dl, args.epochs, score_fn=score_fn, checkpoints=model_name,
                early_stopping=15)

    trainer.model.load_state_dict(torch.load(f'{model_name}.pth'))

    trainer.test(test_dl, score_fn=score_fn)
    