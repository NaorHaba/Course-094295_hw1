import argparse
import pandas as pd
import torch.nn as nn
import torch
import wandb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random

from utils.score_functions import F1
from LSTM.LSTMTrainer import RNNTrainer
from torch.utils.data import DataLoader, WeightedRandomSampler

from LSTM.patient_LSTM import PatientDataset, patientLSTM, batch_collate

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
    remove_healthy = random.sample(healthy, int((1 - under_sample_rate) * len(healthy)))
    train = train[~train['id'].isin(remove_healthy)]
    return train


def weighted_over_sampler(train):
    samples_weight = []
    healthy = set(train[train['SepsisLabel'] == 0.0]['id'].unique())
    sick = set(train[train['SepsisLabel'] == 1.0]['id'].unique())
    for patient, label in train.groupby('id')['SepsisLabel'].max().items():
        if label == 1.0:
            samples_weight.append(1. / len(sick))
        else:
            samples_weight.append(1. / len(healthy))
    samples_weight = torch.tensor(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    return sampler


def scaler_fn(scaler, scaling_columns, data, test=False):
    if not test:
        data[scaling_columns] = scaler.fit_transform(data[scaling_columns])
    else:
        data[scaling_columns] = scaler.transform(data[scaling_columns])
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and Validate model')

    # Paths
    parser.add_argument('--train_file', type=str, help='path to train csv file',
                        default='../data/train_raw.csv')
    parser.add_argument('--test_file', type=str, help='path to test csv file',
                        default='../data/test_raw.csv')
    parser.add_argument('--valid_size', type=float, help='validation data proportion for split',
                        default=0.15)
    # Data parameters
    parser.add_argument('--sampling_method', type=str, help='method for sampling (over or under sampling)',
                        default='over', choices=['under', 'over', 'under-over'])
    parser.add_argument('--under_sample_rate', type=float, help='sampling rate for sampling method',
                        default=0.5)
    parser.add_argument('--remove_columns', type=str, help='method for columns removal',
                        default='')
    parser.add_argument('--scale_method', type=str, help='method for scaling (standard or minmax scalers)',
                        default='minmax', choices=['standard', 'minmax'])
    parser.add_argument('--scaling_columns', type=str, help='method for columns removal',
                        default='')  # '_' delimited: "HR_Temp_..."
    # Model parameters
    parser.add_argument('--train_batch_size', type=int, help='train batch size',
                        default=8)
    parser.add_argument('--test_batch_size', type=int, help='test batch size',
                        default=32)
    parser.add_argument('--window_size', type=int, help='window size',
                        default=30)
    parser.add_argument('--hidden_dim', type=int, help='',
                        default=256)
    parser.add_argument('--LSTM_n_layers', type=int, help='',
                        default=2)
    parser.add_argument('--dropout', type=float, help='',
                        default=0.3)
    parser.add_argument('--lr', type=float, help='',
                        default=0.005)
    parser.add_argument('--true_threshold', type=float, help='',
                        default=0.5)
    parser.add_argument('--epochs', type=int, help='',
                        default=5)
    parser.add_argument('--logging_mode', type=str, choices=['online', 'offline', 'disabled'], help='',
                        default='online')

    args = parser.parse_args()

    wandb.login()
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
                               'ICULOS']
        else:
            scaling_columns = args.scaling_columns.split('_')
        train = scaler_fn(scaler, scaling_columns, train)
        valid = scaler_fn(scaler, scaling_columns, valid, test=True)
        test = scaler_fn(scaler, scaling_columns, test, test=True)

    # create data loaders
    train_X, train_y = split_features_label(train)
    valid_X, valid_y = split_features_label(valid)
    test_X, test_y = split_features_label(test)

    train_ds = PatientDataset(train_X, train_y, args.window_size)
    valid_ds = PatientDataset(valid_X, valid_y, args.window_size)
    test_ds = PatientDataset(test_X, test_y, args.window_size)

    if sampler is not None:
        train_dl = DataLoader(train_ds, batch_size=args.train_batch_size, collate_fn=batch_collate, sampler=sampler)
    else:
        train_dl = DataLoader(train_ds, batch_size=args.train_batch_size, collate_fn=batch_collate, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=args.test_batch_size, collate_fn=batch_collate)
    test_dl = DataLoader(test_ds, batch_size=args.test_batch_size, collate_fn=batch_collate)

    # build model
    model = patientLSTM(features_dim=train_X.shape[1] - 1, hidden_dim=args.hidden_dim, n_layers=args.LSTM_n_layers,
                        dropout=args.dropout)

    loss_fn = nn.BCEWithLogitsLoss()
    score_fn = F1()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    trainer = RNNTrainer(model, loss_fn, optimizer, true_threshold=args.true_threshold)

    # train and test
    with wandb.init(project='hw1', entity='course094295', mode=args.logging_mode, config=vars(args)):
        model_name = f'../{wandb.run.dir}/model.pth'
        trainer.fit(train_dl, valid_dl, args.epochs, score_fn=score_fn, checkpoints=model_name,
                    early_stopping=15)
        wandb.watch(model, log_freq=100)

        trainer.model.load_state_dict(torch.load(model_name)['model_state'])

        trainer.test(test_dl, score_fn=score_fn)
