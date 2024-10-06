import pandas as pd
import numpy as np
import gc
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

def load_data(filepath):
    df = pd.read_csv(filepath, header=None)

    column_headers = [
        "age",
        'employment_type',
        'weighting_factor',
        'level_of_education',
        'training_period',
        'marital_status',
        'employment_area',
        'partnership',
        'ethnicity',
        'gender',
        'gains_financial',
        'losses_financial',
        'weekly_working_time',
        'country',
        'income'
    ]

    df.rename(columns=dict(zip(df.columns, column_headers)), inplace=True)

    return df

def correct_erroneous(df):
    df['country'] = df['country'].replace({
        'South': 'Unknown',
        ' Trinadad&Tobago': 'Trinidad&Tobago',
        ' Columbia': 'Colombia',
    })

    return df

def correct_missing(df):
    def impute_missing_values(column_name):
        mode_value = df[column_name].mode()[0]
        df[column_name] = df[column_name].replace(" ?", mode_value)

    impute_missing_values('employment_type')
    impute_missing_values('employment_area')

    df['country'] = df['country'].replace(" ?", "Unknown")
    df['income'] = df['income'].replace(" ?", "Unknown")

    return df

def separate_target(df):
    target = df['income']
    df = df.drop('income', axis=1)    
    return df, target

def hot_encoding(df):
    df = pd.get_dummies(df, drop_first=True)
    return df

def select_labeled_data(df, target_column='income', missing_value='Unknown'):
    df = df[df[target_column] != missing_value]
    return df

def preprocess(df):
    df = correct_erroneous(df)
    df = correct_missing(df)
    df = select_labeled_data(df)
    print(f"Initial Training Set Size: {len(df)}")

    df, target = separate_target(df)

    df_resample1, target_resample1 = resample_df(df, target, 'employment_type', [' Private'])
    df_resample2, target_resample2 = resample_df(df, target, 'ethnicity', [' White'])
    df_resample3, target_resample3 = resample_df(df, target, 'gender', [' Male'])
    df_resample4, target_resample4 = resample_df(df, target, 'country', [' United-States'])
    df_resample5, target_resample5 = resample_df(df, target, 'marital_status', [' Married-civ-spouse', ' Never-married'])

    df = pd.concat([df_resample1, df_resample2, df_resample3, df_resample4, df_resample5])
    target = pd.concat([target_resample1, target_resample2, target_resample3, target_resample4, target_resample5])

    del df_resample1, df_resample2, df_resample3, df_resample4, df_resample5
    del target_resample1, target_resample2, target_resample3, target_resample4, target_resample5

    df = apply_log_normalization(df, 'gains_financial')
    df = apply_log_normalization(df, 'losses_financial')

    df = hot_encoding(df)

    gc.collect()

    return df, target

def split_data(df, target, test_size=0.15, val_size=0.15, random_state=42):
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    target = target.sample(frac=1, random_state=random_state).reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=test_size, random_state=random_state, stratify=target)
    
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size after test split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size_adjusted, random_state=random_state, stratify=y_train)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def resample_df(X, y, category, value, random_state=42):
    data = pd.concat([X, y], axis=1)

    if isinstance(value, list) and len(value) == 2:
        matching_values = data[(data[category] != value[0]) & (data[category] != value[1])]
        other_values = data[(data[category] == value[0]) | (data[category] == value[1])]
    elif len(value) == 1:
        value = value[0]
        matching_values = data[data[category] == value]
        other_values = data[data[category] != value]
    else:
        raise KeyError('The provided list with categories to resample is not acceptable')

    other_values_resampled = resample(
        other_values, 
        replace=True, 
        n_samples=len(matching_values),
        random_state=random_state
    )

    data_resampled = pd.concat([matching_values, other_values_resampled])

    X_resampled = data_resampled.drop(columns=y.name)
    y_resampled = data_resampled[y.name]

    return X_resampled, y_resampled


def apply_log_normalization(df, column):
    df[column] = np.log1p(df[column])
    
    return df