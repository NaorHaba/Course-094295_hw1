import argparse
import pickle

import pandas as pd
import torch

from LSTM.patient_LSTM import patientLSTM, PatientDataset, batch_collate
from LSTM.train_deploy import scaler_fn, split_features_label


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
    test_ds = PatientDataset(test_X, test_y, 28)

    # build and load model
    model = patientLSTM(features_dim=42, hidden_dim=47, n_layers=1, dropout=0.57)
    model_name = f'models/LSTM_final_model.pth'
    model.load_state_dict(torch.load(model_name)['model_state'])

    # predict
    results = {'id': [], 'prediction': []}
    batch_size = 32
    for i in range(0, len(test_ds.patients), batch_size):
        batch = []
        ids = []
        for j in range(i, min(i + batch_size, len(test_ds.patients))):
            x, y = test_ds[j]
            batch.append((x, y))
            ids.append(test_ds.patients[j])
        x, _ = batch_collate(batch)
        output = model(x).squeeze(1)
        prediction = torch.sigmoid(output)
        prediction = list((prediction > 0.5).int().numpy())
        results['id'] += ids
        results['prediction'] += prediction

    pd.DataFrame(results).to_csv('prediction.csv', index=False, header=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Read a given file and predict labels using our LSTM model')

    # Paths
    parser.add_argument('--test_file', type=str, help='path to test csv file',
                        default='data/test_raw.csv')

    args = parser.parse_args()
    main(args.test_file)
