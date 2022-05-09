import argparse
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler

from LSTM.train_validate import scaler_fn
from LogisticRegression.train_validate_test_deploy import aggregate_data
from preprocess_data import read_all_data


def main(test_file, model_path):
    # read data
    print('Reading CSV...')
    te_df = pd.read_csv(test_file, index_col=0, dtype='float')
    print('Done')

    te_df['id'] = te_df['id'].astype(int)

    print('Aggregating data...', end=' ')
    X_test, y_test = aggregate_data(te_df)
    print('Done.')

    print('Scaling data...', end=' ')
    scaling_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'Age', 'HospAdmTime', 'ICULOS']
    scaler = StandardScaler()
    X_test = scaler_fn(scaler, scaling_columns, X_test, test=True)
    print('Done.')

    print('Selecting features...', end=' ')
    columns = None  # TODO: complete
    X_test = X_test[columns]
    print('Done.')

    # build and load model
    model = pickle.load(open(model_path, 'rb'))

    # predict
    print('Testing model...')
    y_pred = model.predict(X_test)

    preds = pd.concat([X_test['id'], y_pred])
    preds.to_csv('prediction.csv', index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read a given file and predict labels using our baseline model')
    # Paths
    parser.add_argument('--test_file', type=str, help='path to test csv file',
                        default='../data/test_raw.csv')
    args = parser.parse_args()

    model_path = '../models/LR_final_model.pkl'
    main(args.test_file, model_path)
