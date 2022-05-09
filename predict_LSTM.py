import argparse
import pickle

import pandas as pd
import torch
from torch.utils.data import DataLoader

from LSTM.patient_LSTM import patientLSTM, PatientDataset, batch_collate
from LSTM.train_validate import scaler_fn, split_features_label


def main(test_file):

    # read data
    print('Reading CSV...')
    test = pd.read_csv(test_file, index_col=0, dtype='float')
    print('Done')

    test['id'] = test['id'].astype(int)

    # scale columns
    scaling_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'Age', 'HospAdmTime', 'ICULOS']
    scaler = pickle.load(open('models/LSTM_scaler.pkl', 'rb'))
    test = scaler_fn(scaler, scaling_columns, test, test=True)

    # create dataset
    test_X, test_y = split_features_label(test)
    test_ds = PatientDataset(test, test_y, 35)

    # build and load model
    model = patientLSTM(features_dim=42, hidden_dim=512, n_layers=3, dropout=0.45)
    model_name = f'../models/LSTM_final_model.pth'
    model.load_state_dict(torch.load(model_name)['model_state'])

    # predict
    results = {'id': [], 'prediction': []}
    for i in range(len(test_ds.patients)):
        x, _ = test_ds[i]
        output = model(x).squeeze(1)
        prediction = torch.sigmoid(output)
        prediction = (prediction > 0.5).int()[0].item()
        results['id'].append(test_ds.patients[i])
        results['prediction'].append(prediction)

    pd.DataFrame(results).to_csv('prediction.csv', index=False, header=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Read a given file and predict labels using our LSTM model')

    # Paths
    parser.add_argument('--test_file', type=str, help='path to test csv file',
                        default='data/test_raw.csv')

    args = parser.parse_args()
    main(args.test_file)
