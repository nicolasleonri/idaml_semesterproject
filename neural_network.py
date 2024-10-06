import pandas as pd
import numpy as np
import tensorflow as tf
from utils import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow.keras import Input  # Import the Input layer
from sklearn.metrics import f1_score

def build_model(input_dim, dropout_rate=0.5):
    """Build and compile a neural network model using Keras."""
    model = Sequential()

    model.add(Input(shape=(input_dim,)))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def k_fold_cross_validation(X_train, y_train, n_splits=2, epochs=1, pos_label=' >50K', batch_size=16):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    recalls = []
    best_score = 0

    for train_index, val_index in kf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

        model = build_model(input_dim=X_fold_train.shape[1])
        
        model.fit(X_fold_train, y_fold_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        y_val_pred = (model.predict(X_fold_val) > 0.5).astype(int)

        accuracies.append(accuracy_score(y_fold_val, y_val_pred))
        recalls.append(recall_score(y_fold_val, y_val_pred))

        if accuracies[-1] > best_score:
            best_score = accuracies[-1]
            best_model = model

    average_accuracy = sum(accuracies) / len(accuracies)
    average_recall = sum(recalls) / len(recalls)

    return best_model, average_accuracy, average_recall

def nested_cross_validation(X_train, y_train, X_val, y_val, best_model_architecture, param_grid, epochs=5, batch_size=16):
    best_score = 0
    best_model = None
    best_params = None
    accuracies = []
    recalls = []
    results = []

    for params in param_grid:
        # Build model with the best architecture but with new hyperparameters
        model = tf.keras.models.clone_model(best_model_architecture)  # Clone the architecture
        model.set_weights(best_model_architecture.get_weights())  # Copy weights

        # Adjust dropout rate if specified
        for layer in model.layers:
            if isinstance(layer, Dropout):
                layer.rate = params.get('dropout_rate', 0.5)  # Update dropout rate

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compile the model
        
        # Train the model
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Evaluate on validation set
        y_val_pred = (model.predict(X_val) > 0.5).astype(int)
        accuracies.append(accuracy_score(y_val, y_val_pred))
        recalls.append(recall_score(y_val, y_val_pred))

        if accuracies[-1] > best_score:
            best_score = accuracies[-1]
            best_model = model
            best_params = params

        results.append({
            'dropout_rate': params.get('dropout_rate', None),  # Adjust based on your actual param names
            'learning_rate': params.get('learning_rate', None),
            'Accuracy': accuracies[-1],
            'Recall': recalls[-1]
        })

    average_accuracy = sum(accuracies) / len(accuracies)
    average_recall = sum(recalls) / len(recalls)

    results_df = pd.DataFrame(results)
    results_df.to_csv('./results/nn_validation_results.csv', index=False)
    plot_results_nn(results_df, './results/nn_validation_results.png')

    return best_model, best_params, average_accuracy, average_recall

def find_optimal_threshold(model, X_val, y_val):
    y_val_probs = model.predict(X_val).flatten()  # Get predicted probabilities
    thresholds = np.arange(0.0, 1.01, 0.01)  # Range of thresholds from 0.0 to 1.0
    best_f1 = 0
    optimal_threshold = 0.5

    for threshold in thresholds:
        y_val_pred = (y_val_probs > threshold).astype(int)  # Apply the threshold
        f1 = f1_score(y_val, y_val_pred)

        if f1 > best_f1:
            best_f1 = f1
            optimal_threshold = threshold

    return optimal_threshold, best_f1

def evaluate_model(model, X_test, y_test, threshold = 0.5):
    """Evaluate the logistic regression model on the test set."""
    y_test_pred = (model.predict(X_test) > threshold).astype(int)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_report = classification_report(y_test, y_test_pred)
    test_confusion = confusion_matrix(y_test, y_test_pred)

    return test_accuracy, test_report, test_confusion


def main():
    df = load_data("./data/einkommen.train")

    df, target = preprocess_logistic(df)

    target = target.apply(lambda x: 1 if x == ' >50K' else 0)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target)

    best_model, average_accuracy, average_recall = k_fold_cross_validation(X_train, y_train)

    print(f"Average K-Fold Accuracy: {average_accuracy:.4f}")
    print(f"Average K-Fold Recall: {average_recall:.4f}")

    param_grid = [
    {'dropout_rate': 0.2, 'learning_rate': 0.001, 'batch_size': 16},
    {'dropout_rate': 0.3, 'learning_rate': 0.001, 'batch_size': 32},
    {'dropout_rate': 0.5, 'learning_rate': 0.001, 'batch_size': 64},
    {'dropout_rate': 0.5, 'learning_rate': 0.0005, 'batch_size': 16},
    {'dropout_rate': 0.5, 'learning_rate': 0.01, 'batch_size': 32},
    ]

    best_model, best_params, average_accuracy, average_recall = nested_cross_validation(X_train, y_train, X_val, y_val, best_model, param_grid)
    
    print(f"Nested-Cross Accuracy: {average_accuracy:.4f}")
    print(f"Nested-Cross Recall: {average_recall:.4f}")
    print(f"Best params: {best_params}")

    test_accuracy, test_report, test_confusion = evaluate_model(best_model, X_test, y_test)

    print("Test Accuracy:", test_accuracy)
    print("Test Classification Report:\n", test_report)
    print("Test Confusion Matrix:\n", test_confusion)

    optimal_threshold, best_f1 = find_optimal_threshold(best_model, X_val, y_val)
    print(f"Optimal Threshold: {optimal_threshold}, Best F1-Score: {best_f1}")

    test_accuracy, test_report, test_confusion = evaluate_model(best_model, X_test, y_test, optimal_threshold)

    y_test_proba = best_model.predict(X_test).flatten()  # Get probabilities for the positive class
    # Convert string labels to numeric labels
    plot_auc(y_test, y_test_proba, './results/roc_curve_nn.png')



    print("Test Accuracy:", test_accuracy)
    print("Test Classification Report:\n", test_report)
    print("Test Confusion Matrix:\n", test_confusion)

    return None

if __name__ == "__main__":
    main()