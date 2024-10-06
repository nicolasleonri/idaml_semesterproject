from utils import *
import gc
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from sklearn.model_selection import ParameterGrid

def k_fold_cross_validation(X_train, y_train, param_grid, pos_label=' >50K', random_state=42):
    """Perform K-Fold cross-validation to find the best Logistic Regression model."""
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    best_score = 0
    best_model = None
    accuracies = []
    recalls = []

    for params in ParameterGrid(param_grid):
        for train_index, val_index in kf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

            model = LogisticRegression(**params, random_state=random_state, max_iter=10000)
            model.fit(X_fold_train, y_fold_train)
            y_k_val_pred = model.predict(X_fold_val)

            accuracies.append(accuracy_score(y_fold_val, y_k_val_pred))
            recalls.append(recall_score(y_fold_val, y_k_val_pred, pos_label=pos_label))

            if accuracies[-1] > best_score:
                best_score = accuracies[-1]
                best_model = model

    average_accuracy = sum(accuracies) / len(accuracies)
    average_recall = sum(recalls) / len(recalls)

    return best_model, average_accuracy, average_recall

def nested_cross_validation(X_train, y_train, X_val, y_val, best_model, param_grid, pos_label=' >50K', random_state=42):
    grid_search = GridSearchCV(
        best_model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv('./results/logistic_regression_grid_search_results.csv', index=False)
    plot_results_logistic(results_df, './results/logistic_validation_hyperparameters.png')


    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred, pos_label=pos_label)

    return best_model, val_accuracy, val_recall

def evaluate_model(model, X_test, y_test):
    """Evaluate the logistic regression model on the test set."""
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_report = classification_report(y_test, y_test_pred)
    test_confusion = confusion_matrix(y_test, y_test_pred)

    return test_accuracy, test_report, test_confusion

def main():
    df = load_data("./data/einkommen.train")

    # df, target = preprocess_trees(df)
    df, target = preprocess_logistic(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target)
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.001, 0.01, ], #0.1, 1, 10, 100
        'penalty': ['l2'],
        'class_weight': ['balanced', None] #, 
    }

    best_model, average_accuracy, average_recall = k_fold_cross_validation(X_train, y_train, param_grid)

    print(f"Average K-Fold Accuracy: {average_accuracy:.4f}")
    print(f"Average K-Fold Recall: {average_recall:.4f}")

    best_model, average_accuracy, average_recall = nested_cross_validation(X_train, y_train, X_val, y_val, best_model, param_grid)

    print(f"Nested-Cross Accuracy: {average_accuracy:.4f}")
    print(f"Nested-Cross Recall: {average_recall:.4f}")

    test_accuracy, test_report, test_confusion = evaluate_model(best_model, X_test, y_test)
        
    y_test_proba = best_model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
    y_test_numeric = [1 if label == ' >50K' else 0 for label in y_test]  # Mapping ' >50K' to 1 and ' <=50K' to 0
    plot_auc(y_test_numeric, y_test_proba, './results/roc_curve_logreg.png')


    print("Test Accuracy:", test_accuracy)
    print("Test Classification Report:\n", test_report)
    print("Test Confusion Matrix:\n", test_confusion)

    return None

if __name__ == "__main__":
    main()