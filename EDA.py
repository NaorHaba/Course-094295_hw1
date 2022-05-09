from functools import partial

import numpy as np
import pandas as pd
import os

TRAIN_PATH = 'data/train'
TEST_PATH = 'data/test'


def read_all_data(folder):
    files = []
    for file in os.listdir(folder):
        if file.endswith('psv'):
            _id = int(file.replace('.psv', '').split('_')[1])
            df = pd.read_csv(os.path.join(folder, file), sep='|')
            df['id'] = _id
            files.append(df)
    return pd.concat(files, ignore_index=True)


def add_ICULOS_rows(g, fill_cols):
    # add missing ICULOS values and bfill desired columns
    # fill_cols - list of column groups
    first_hour = g.ICULOS.iloc[0]
    if first_hour != 1:
        ICULOS_hours = {'ICULOS': list(range(1, first_hour))}
        g = pd.concat([pd.DataFrame(ICULOS_hours), g])
        for col_group in fill_cols:
            g[col_group] = g[col_group].bfill()
    return g


def add_unknown_unit_column(df):
    df['Unit3'] = 0.0
    df.loc[df.Unit1.isna(), 'Unit3'] = 1.0  # we know Unit2 is also nan when Unit1 is nan
    df[['Unit1', 'Unit2']] = df[['Unit1', 'Unit2']].fillna(0.0)
    return df


def fix_and_smooth(df, fix_cols, com=0.5):
    # smooth and fix range of desired columns
    # fix_cols = dictionary of columns that need fixing, and range in case it needs to be changed
    id_df = df.groupby('id')
    for col, (low, high) in fix_cols.items():
        if low and high:
            df.loc[(df[col] >= high) | (df[col] <= low), col] = np.nan
        df[col] = id_df[col].ewm(com=com).mean().bfill().ffill().reset_index(level=0, drop=True)
    return df


def engineer_lab_values(df, lab_values):
    df[lab_values] = df[lab_values].notna().astype(int)  # marking all NAN as 0 and values as 1
    # summing to get the amount of tests for each column than dividing by hours passed so far
    df[lab_values] = df[list(lab_values) + ['id']].groupby('id').cumsum().divide(df['ICULOS'], axis=0)
    return df[lab_values]


def pipeline_eda(t_df, columns, save_path, test=False, icu_fill_cols=None, fix_smooth_cols=None, lab_values=False):
    # add ICU where it's missing and bfill
    if icu_fill_cols:
        print('Filling ICULOS')
        t_df = t_df.sort_values(['id', 'ICULOS']).groupby('id').apply(lambda g: add_ICULOS_rows(g, icu_fill_cols)) \
            .reset_index(drop=True)

    # add Unit3 for unknown Unit1/Unit2
    print('Adding Unit3 column')
    t_df = add_unknown_unit_column(t_df)

    # smooth data and fix range in case of unreasonable values (remove extreme outliers)
    # smooth all relevant data in one function
    if fix_smooth_cols:
        print('Fixing and smoothing data')
        t_df = fix_and_smooth(t_df, fix_smooth_cols)

    # engineer lab values as frequencies
    if lab_values:
        print('Engineering lab values')
        t_df[columns['lab_values']] = engineer_lab_values(t_df, columns['lab_values'])
    else:
        print('Removing lab values')
        t_df.drop(columns=columns['lab_values'])

    # encode categorical features
    print('Encoding categoricals')
    t_df = pd.get_dummies(t_df, columns=['Gender'])

    # filter y
    if test:  #TODO: add ass argument
        print('Filtering SepsisLabel')
        t_df_raw = t_df[~((t_df['SepsisLabel'] == 1.0) & (t_df.groupby('id')['SepsisLabel'].diff() == 0.0))]

    # if test:
    #     patients_labels = (t_df_raw.groupby('id').max()['SepsisLabel'] > 0).astype(int).to_dict()
    #     t_df_raw['PatientLabel'] = np.zeros(len(t_df_raw))
    #     t_df_raw['PatientLabel'] = t_df_raw['id'].apply(lambda l: 1 if patients_labels[l] == 1 else 0)

    print('Saving file')
    t_df_raw.to_csv(save_path, index=False)

    return t_df_raw


if __name__ == '__main__':
    #TODO: add argparse
    t_df = read_all_data(TEST_PATH)

    columns = {'vital_signs': t_df.columns[:8], 'lab_values': t_df.columns[8: 34],
               'demographics': t_df.columns[34: 40], 'outcome': t_df.columns[40], '_id': t_df.columns[-1]}

    icu_fill_cols = [columns['demographics'], columns['outcome'], columns['_id']]

    fix_smooth_cols = {'HR': (None, None),
                'O2Sat': (None, None),
                'Temp': (30, 43),
                'SBP': (None, None),
                'MAP': (None, None),
                'DBP': (1, 250),
                'Resp': (None, None),
                'EtCO2': (None, None),
                }

    pipeline_eda(t_df, columns,
                 test=True,
                 save_path='data/test_raw.csv',
                 icu_fill_cols=icu_fill_cols,
                 fix_smooth_cols=fix_smooth_cols,
                 lab_values=columns['lab_values'])
