import pickle

import pandas as pd
import random

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from LSTM.train_deploy import scaler_fn


def read_data(file):
    df = pd.read_csv(file, index_col=0, dtype='float')
    df['id'] = df['id'].astype(int)
    return df


def aggregate_data(df, hours=30):
    # take last #hours of each patient and aggregate (mean or max)
    # then scale values
    max_cols = ['ICULOS', 'SepsisLabel']
    mean_cols = [col for col in df if col not in ['id'] + max_cols]
    df = pd.concat([
        df.groupby('id')[mean_cols].apply(lambda x: x.tail(hours).mean()),
        df.groupby('id')[max_cols].apply(lambda x: x.tail(hours).max())], axis=1)
    y = df['SepsisLabel']
    X = df.drop(['SepsisLabel'], axis=1)
    return X, y


def select_features(X_t, y_t):
    sfs = SequentialFeatureSelector(LogisticRegression(max_iter=10000), n_features_to_select=0.5, scoring='f1', n_jobs=-1)
    sfs = sfs.fit(X_t, y_t)
    chosen_numeric = X_t.columns[sfs.support_]
    print('Chosen numeric columns:', chosen_numeric)
    X_t = pd.DataFrame(sfs.transform(X_t), columns=chosen_numeric, index=X_t.index)
    return X_t, chosen_numeric


if __name__ == '__main__':
    random.seed(42)
    train_file = 'data/train_raw.csv'
    test_file = 'data/test_raw.csv'

    print('Reading train file...', end=' ')
    tr_df = read_data(train_file)
    te_df = read_data(test_file)
    print('Done.')

    print('Aggregating data...', end=' ')
    X_train, y_train = aggregate_data(tr_df)
    X_test, y_test = aggregate_data(te_df)
    print('Done.')

    print('Scaling data...', end=' ')
    scaling_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'Age', 'HospAdmTime', 'ICULOS']
    scaler = StandardScaler()
    X_train = scaler_fn(scaler, scaling_columns, X_train)
    X_test = scaler_fn(scaler, scaling_columns, X_test, test=True)
    pickle.dump(scaler, open('models/LR_scaler.pkl', 'wb'))
    print('Done.')

    print('Selecting features...', end=' ')
    X_train, support = select_features(X_train, y_train)
    X_test = X_test[support]
    print('Done.')

    print('Training with cv validation a Logistic Regression model:')
    cv = 5
    max_iter = 10000
    print(f'\tcross validation {cv}, max iterations {max_iter}')
    clf = LogisticRegressionCV(cv=cv, random_state=0, scoring='f1', max_iter=max_iter).fit(X_train, y_train)
    print('Done.')

    print(f'Train F1 Score: {round(f1_score(y_train, clf.predict(X_train)), 3)}')

    print('Saving model...', end=' ')
    filename = 'models/LR_final_model.pkl'
    pickle.dump(clf, open(filename, 'wb'))
    print('Done.')

    print('Testing model...')
    print(f'Test F1 Score: {round(f1_score(y_test, clf.predict(X_test)), 3)}')
