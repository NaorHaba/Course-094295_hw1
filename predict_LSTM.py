import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

from LSTM.patient_LSTM import patientLSTM
from LSTM.train_validate import scaler_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read a given file and predict labels using our LSTM model')

    # Paths
    parser.add_argument('--test_file', type=str, help='path to test csv file',
                        default='../data/test_raw.csv')
    # Data parameters

    args = parser.parse_args()

    # read data
    print('Reading CSV...')
    test = pd.read_csv(args.test_file, index_col=0, dtype='float')
    print('Done')

    test['id'] = test['id'].astype(int)

    # scale columns
    scaling_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'Age', 'HospAdmTime', 'ICULOS']
    test = scaler_fn(scaler, scaling_columns, test, test=True)

    # create data loader
    test_ds = TestDataset(test_X, test_y, 35)

    test_dl = DataLoader(test_ds, batch_size=args.test_batch_size, collate_fn=batch_collate)

    # build and load model
    model = patientLSTM(features_dim=42, hidden_dim=512, n_layers=3, dropout=0.45)
    model_name = f'../models/LSTM_final_model.pth'
    model.load_state_dict(torch.load(model_name)['model_state'])

    # predict