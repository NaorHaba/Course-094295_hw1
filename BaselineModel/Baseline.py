import pickle

import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


def read_data(file):
    df = pd.read_csv(file, index_col=0, dtype='float')
    df['id'] = df['id'].astype(int)
    return df


def split_train_validation(tr_df, train_size=0.95):
    sick = set(tr_df[tr_df['SepsisLabel'] == 1.0]['id'].unique())
    healthy = set(tr_df['id'].unique()) - sick
    t_sick = set(random.sample(tuple(sick), int(train_size*len(sick))))
    v_sick = sick - t_sick
    t_healthy = set(random.sample(tuple(healthy), int(train_size*len(healthy))))
    v_healthy = healthy - t_healthy
    train = tr_df[tr_df.id.isin(list(t_sick) + list(t_healthy))].sort_values(['id', 'SepsisLabel'])
    valid = tr_df[tr_df.id.isin(list(v_sick) + list(v_healthy))].sort_values(['id', 'SepsisLabel'])
    return train, valid


def aggregate_and_scale(df, scaling_columns, scaler, hours=7):
    # take last #hours of each patient and aggregate (mean or max)
    # then scale values
    max_cols = ['ICULOS', 'SepsisLabel']
    mean_cols = [col for col in df if col not in ['id'] + max_cols]
    X_t = pd.concat([df.groupby('id')[mean_cols].apply(
        lambda x: x.tail(hours).mean()), df.groupby('id')[max_cols].apply(lambda x: x.tail(hours).max())], axis=1)
    X_t[scaling_columns] = scaler.fit_transform(X_t[scaling_columns])  # TODO: return scaler and use it on test data
    if 'SepsisLabel' in X_t.columns:
        y_t = X_t['SepsisLabel']
        X_t = X_t.drop(['SepsisLabel'], axis=1)
        return X_t, y_t
    else:
        return X_t


def select_features(X_t, y_t):
    sfs = SequentialFeatureSelector(LogisticRegression(), n_features_to_select=0.5, scoring='f1', n_jobs=-1)
    sfs = sfs.fit(X_t, y_t)
    chosen_numeric = X_t.columns[sfs.support_]
    print('Chosen numeric columns:', chosen_numeric)
    X_t = pd.DataFrame(sfs.transform(X_t), columns=chosen_numeric, index=X_t.index)
    return X_t, sfs.support_


if __name__ == '__main__':
    random.seed(42)
    print('Reading train file...', end=' ')
    train_file = '../data/train_raw.csv'
    tr_df = read_data(train_file)
    print('Done.')

    print('Aggregating and scaling...', end=' ')
    scaling_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'Age', 'HospAdmTime', 'ICULOS']
    scaler = MinMaxScaler()
    X_train, y_train = aggregate_and_scale(tr_df, scaling_columns, scaler)
    print('Done.')

    print('Selecting features...', end=' ')
    X_train, support = select_features(X_train, y_train)
    save_path = '../data/X_train_LR.csv'
    X_train.to_csv(f'../data/{save_path}')
    print(f'Saved data for training in {save_path}')

    print('Training Logistic Regression model:')
    cv = 5
    max_iter = 10000
    print(f'\tcross validation {cv}, max iterations {max_iter}')
    clf = LogisticRegressionCV(cv=cv, random_state=0, scoring='f1', max_iter=max_iter).fit(X_train, y_train)
    print('Done.')

    print(f'F1 Score: {round(clf.score(X_train, y_train), 3)}')

    print('Saving model...', end=' ')
    filename = '../models/LR_final_model.pkl'
    pickle.dump(clf, open(filename, 'wb'))
    print('Done.')

    print('Testing model...')

    print('Reading data...', end=' ')
    test_file = 'data/test_raw.csv'
    te_df = read_data(test_file)
    print('Done.')

    print('Aggregating and scaling...', end=' ')
    X_test = aggregate_and_scale(te_df, scaling_columns, scaler)
    print('Done.')
    print('Saving data with chosen columns...', end=' ')
    chosen_numeric = X_test.columns[support]
    X_test = X_test[chosen_numeric]
    save_path = '../data/X_test_LR.csv'
    X_test.to_csv(f'../data/{save_path}')
    print(f'Saved data for testing in {save_path}')

    print('Predicting...')
    y_pred = clf.predict(X_test)
    print()
    print('Finished.')