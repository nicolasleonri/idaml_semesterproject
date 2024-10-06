import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import gc
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fancyimpute import IterativeImputer  # EM-based imputation
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc

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

    missing_values_table(df)
    save_continuous_summary(df)

    return df

def plot_results_trees(df, filepath, flag=None):
    selected_columns = df[['param_min_samples_split', 'param_max_depth', 'mean_test_score']]

    if flag != None:
        selected_columns = selected_columns.groupby(['param_min_samples_split', 'param_max_depth',], as_index=False).mean()

    heatmap_data = selected_columns.pivot(
        index='param_min_samples_split', 
        columns='param_max_depth', 
        values='mean_test_score'
    )

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 6))

    # Create the heatmap
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Mean Test Score'})

    # Add labels and title
    plt.title('Validation Accuracy Heatmap for Decision Tree Hyperparameters')
    plt.xlabel('Max Depth')
    plt.ylabel('Min Samples Split')
    #plt.show()

    plt.savefig(filepath)

def plot_results_logistic(df, filepath):
    # Select relevant columns
    selected_columns = df[['param_C', 'param_class_weight', 'param_penalty', 'mean_test_score']]

    heatmap_data = selected_columns.pivot(
        index='param_C', 
        columns='param_class_weight', 
        values='mean_test_score'
    )
    # Create the heatmap
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Mean Test Score'})

    # Add labels and title
    plt.title('Validation Accuracy Heatmap for Decision Tree Hyperparameters')
    plt.xlabel('Param C')
    plt.ylabel('Class weight')
    #plt.show()

    plt.savefig(filepath)

def plot_results_nn(df, filepath):
    # Select relevant columns
    selected_columns = df[['dropout_rate', 'learning_rate', 'Accuracy', 'Recall']]

    heatmap_data = selected_columns.pivot(
        index='dropout_rate', 
        columns='learning_rate', 
        values='Accuracy'
    )

    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Mean Test Score'})

    # Add labels and title
    plt.title('Validation Accuracy Heatmap for Decision Tree Hyperparameters')
    plt.xlabel('Dropout rate')
    plt.ylabel('Learning rate')
    plt.savefig(filepath)

def plot_auc(y_true, y_scores, filepath):
    """
    Calculate and plot the ROC curve and AUC for binary classification.
    
    Parameters:
    - y_true: Array-like, true binary labels.
    - y_scores: Array-like, predicted probabilities for the positive class.
    - filepath: Path to save the AUC plot.
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Calculate AUC
    roc_auc = auc(fpr, tpr)

    # Set up the plot
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc='lower right')
    
    # Save the plot
    plt.savefig(filepath)
    plt.close()  # Close 


def correct_erroneous(df):
    df['country'] = df['country'].replace({
        # 'South': 'Unknown',
        ' Trinadad&Tobago': 'Trinidad&Tobago',
        ' Columbia': 'Colombia',
    })

    df = df.drop('weighting_factor', axis=1)

    return df

def em_impute_specific_columns(df, columns_to_impute, missing_value=' ?'):
    df[columns_to_impute] = df[columns_to_impute].replace(missing_value, np.nan)

    label_encoders = {}
    for column in columns_to_impute:
        if df[column].dtype == 'object':  # Only encode if it's a categorical column
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column].astype(str))

    # Apply the EM algorithm using IterativeImputer on the specified columns
    imputer = IterativeImputer(max_iter=10, random_state=0)

    # Impute only the specified columns
    df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

    # Reverse the label encoding to restore original categorical values
    for column in columns_to_impute:
        if column in label_encoders:  # Only reverse if encoding was applied
            df[column] = label_encoders[column].inverse_transform(df[column].astype(int))

    return df

def impute_missing_values(df, column_name):
        mode_value = df[column_name].mode()[0]
        df[column_name] = df[column_name].replace(" ?", mode_value)
        return df

def correct_missing(df):
    df = impute_missing_values(df, 'employment_area')
    df = em_impute_specific_columns(df, ['country', 'employment_type'])
    df['income'] = df['income'].replace(" ?", "Unknown")

    if (df == " ?").sum().sum() > 0:
        raise KeyError("Still got some missing values, pal!")

    return df

def save_continuous_summary(df, flag="prior/", output_dir='./csv'):
    output_dir = os.path.join(output_dir, flag)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f'continous_summary.csv')

    if os.path.exists(file_path):
        return None

    # Select only continuous (numeric) columns
    continuous_columns = df.select_dtypes(include=[np.number])

    # Generate summary statistics for continuous variables
    summary = continuous_columns.describe()

    # Save the summary to a CSV file
    summary.to_csv(file_path)
    print(f"Summary of continuous variables saved to {file_path}")

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

def get_categorical_columns_names(df):
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    categorical_columns = list(set(cols) - set(num_cols))
    return categorical_columns

def plot_class_distributions(df, categorical_columns, flag="prior/", output_dir='./plots/class_distributions'):
    output_dir = os.path.join(output_dir, flag)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sns.set(style="whitegrid")

    for column in categorical_columns:
        file_path = os.path.join(output_dir, f'class_distribution_{column}.png')
        csv_file_path = os.path.join(output_dir, f'class_distribution_{column}.csv')

        # Check if the plot already exists; if so, skip this column
        if os.path.exists(file_path) and os.path.exists(csv_file_path):
            continue

        plt.figure(figsize=(10, 6))

        count_data = df[column].value_counts()

        sns.barplot(x=count_data.index, y=count_data.values, palette="viridis")
        
        plt.title(f'Class Distribution of {column}', fontsize=16)
        plt.xlabel(column, fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.xticks(rotation=45)
        
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        plt.savefig(file_path)
        plt.close()

        # Save the distribution data as CSV
        count_data = df[column].value_counts()
        total_count = count_data.sum()
        proportions = count_data / total_count

        count_data_df = count_data.reset_index()
        count_data_df.columns = [column, 'Frequency']
        count_data_df['Proportion'] = proportions.values  # Add proportions to the DataFrame
        count_data_df.to_csv(csv_file_path, index=False)

        print(f"File {file_path} created.")
        print(f"File {csv_file_path} created.")


def save_class_ratios(df, categorical_columns, flag="prior/", output_file='./csv/'):
    output_dir = os.path.join(output_file, flag)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, flag[:-1])
    output_file = os.path.join(output_dir, "_ratios_categorical.csv")
    
    if os.path.exists(output_file):
        return None

    class_ratios = []

    for column in categorical_columns:
        # Count the occurrences of each category
        count_data = df[column].value_counts()

        # Identify the majority and minority classes
        majority_class_size = count_data.max()  # The size of the largest class
        minority_class_size = count_data.min()  # The size of the smallest class

        # Calculate the ratio of minority class to majority class
        ratio = minority_class_size / majority_class_size

        class_ratios.append({
            'Column': column,
            'Majority Class Size': majority_class_size,
            'Minority Class Size': minority_class_size,
            'Minority-to-Majority Ratio': ratio
        })

    ratios_df = pd.DataFrame(class_ratios)

    ratios_df.to_csv(output_file, index=False)
    print(f"Class ratios saved to {output_file}")

def missing_values_table(df, missing_value=' ?', output_file='./csv/missing_values.csv'):
    missing_data = []

    if not os.path.exists('./csv'):
        os.makedirs('./csv')

    if os.path.exists(output_file):
        return None

    for column in df.columns:
        # Count occurrences of the missing value in the column
        missing_count = (df[column] == missing_value).sum()

        # Calculate the percentage of missing values in the column
        total_count = len(df)
        missing_percentage = (missing_count / total_count) * 100

        if missing_count > 0:
            missing_data.append({
                'Column': column,
                'Missing Count': missing_count,
                'Missing Percentage': missing_percentage
            })

    # Create a DataFrame from the collected missing values data
    missing_df = pd.DataFrame(missing_data)

    # Save the DataFrame to a CSV file
    missing_df.to_csv(output_file, index=False)
    print(f"Missing values table saved to {output_file}")

def preprocess_trees(df):
    df = correct_erroneous(df)
    df = correct_missing(df)
    df = select_labeled_data(df)
    df, target = separate_target(df)

    plot_class_distributions(df, get_categorical_columns_names(df))
    save_class_ratios(df, get_categorical_columns_names(df))

    print(f"Initial Training Set Size: {len(df)}")

    categorical_features_to_amplify = ['employment_type', 'gender']  # Specify your target categorical features

    smote = SMOTENC(categorical_features=get_categorical_columns_names(df), random_state=42)
    df, target = smote.fit_resample(df, target)

    plot_class_distributions(df, get_categorical_columns_names(df), "posterior/")
    save_class_ratios(df, get_categorical_columns_names(df), "posterior/")

    print(f"Resampled Training Set Size: {len(df)}")

    df = apply_log_normalization(df, 'gains_financial')
    df = apply_log_normalization(df, 'losses_financial')
    save_continuous_summary(df, "posterior/")


    df = hot_encoding(df)

    gc.collect()

    return df, target

def preprocess_logistic(df):
    df = correct_erroneous(df)
    df = correct_missing(df)
    df = select_labeled_data(df)
    df, target = separate_target(df)

    print(f"Initial Training Set Size: {len(df)}")

    print(f"Resampled Training Set Size: {len(df)}")

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
    df.loc[:,[column]] = np.log1p(df.loc[:,[column]].astype(float))
    
    return df