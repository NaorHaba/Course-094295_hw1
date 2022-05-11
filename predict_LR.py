import argparse
import pickle

import pandas as pd

from LSTM.train_deploy import scaler_fn
from LogisticRegression.train_deploy import aggregate_data


def main(test_file):
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
    scaler = pickle.load(open('models/LR_scaler.pkl', 'rb'))
    X_test = scaler_fn(scaler, scaling_columns, X_test, test=True)
    print('Done.')

    print('Selecting features...', end=' ')
    columns = ['HR', 'O2Sat', 'Temp', 'Resp', 'BaseExcess', 'HCO3', 'FiO2', 'SaO2',
               'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate',
               'Magnesium', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'Fibrinogen', 'Unit1',
               'ICULOS']
    X_test = X_test[columns]
    # add index
    X_test.index = X_test.index.set_names(['id'])
    X_test = X_test.reset_index()
    print('Done.')

    # build and load model
    model = pickle.load(open('models/LR_final_model.pkl', 'rb'))

    # predict
    print('Testing model...')
    y_pred = pd.Series(model.predict(X_test.drop('id', axis=1)), index=X_test.index)

    preds = pd.concat([X_test['id'], y_pred], axis=1)
    preds.to_csv('LR_prediction.csv', index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read a given file and predict labels using our baseline model')
    # Paths
    parser.add_argument('--test_file', type=str, help='path to test csv file',
                        default='data/test_raw.csv')
    args = parser.parse_args()

    main(args.test_file)
