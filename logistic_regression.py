from utils import *
import gc
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from sklearn.model_selection import GridSearchCV

def train_decision_tree(X_train, y_train, k_folds=5, max_depth=None, min_samples_split=2, pos_label=' >50K', random_state=42):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    accuracies = []
    recalls = []

    best_score = 0
    best_model = None

    for train_index, val_index in kf.split(X_train):
        X_k_train, X_k_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_k_train, y_k_val = y_train.iloc[train_index], y_train.iloc[val_index]
        
        decision_tree = DecisionTreeClassifier(
            criterion='entropy',  # Loss function: Gini impurity
            max_depth=max_depth,  # Regularization: Limit tree depth
            min_samples_split=min_samples_split,  # Regularization: Min samples to split
            random_state=random_state
        )

        decision_tree.fit(X_k_train, y_k_train)

        y_k_val_pred = decision_tree.predict(X_k_val)

        accuracies.append(accuracy_score(y_k_val, y_k_val_pred))

        if accuracies[-1] > best_score:
            best_score = accuracies[-1]
            best_model = decision_tree

        recalls.append(recall_score(y_k_val, y_k_val_pred, pos_label=pos_label))

    average_accuracy = sum(accuracies) / len(accuracies)
    average_recall = sum(recalls) / len(recalls)

    return best_model, average_accuracy, average_recall

def nested_cross_validation(X_train, y_train, X_val, y_val, param_grid, best_model, pos_label=' >50K', random_state=42):
    grid_search = GridSearchCV(
        best_model,
        param_grid,
        cv=5,  # Inner cross-validation folds
        scoring='accuracy',
        n_jobs=-1  # Use all available cores
    )

    # Fit the model on the training set
    grid_search.fit(X_train, y_train)
    
    # Best model from grid search
    best_model = grid_search.best_estimator_

    y_val_pred = best_model.predict(X_val)

    val_accuracy = best_model.score(X_val, y_val)
    val_recall = classification_report(y_val, y_val_pred, output_dict=True)[pos_label]['recall']

    return best_model, val_accuracy, val_recall

def main():
    df = load_data("./data/einkommen.train")

    df, target = preprocess(df)

    print(f"Resampled Training Set Size: {len(df)}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target)
 
    decision_tree_model, average_accuracy, average_recall = train_decision_tree(X_train, y_train, k_folds=5, max_depth=5, min_samples_split=10)
    
    print(f"Average K-Fold Accuracy: {average_accuracy:.4f}")
    print(f"Average K-Fold Recall: {average_recall:.4f}")

    param_grid = {
        'max_depth': [None, 5, 10, 15],  # None means no limit
        'min_samples_split': [2, 5, 10]
    }

    best_decision_tree_model, average_accuracy, average_recall = nested_cross_validation(X_train, y_train, X_val, y_val, param_grid, decision_tree_model)
    print(f"Nested-Cross Accuracy: {average_accuracy:.4f}")
    print(f"Nested-Cross Recall: {average_recall:.4f}")

    # Final evaluation on the test set
    y_test_pred = best_decision_tree_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_report = classification_report(y_test, y_test_pred)
    test_confusion = confusion_matrix(y_test, y_test_pred)

    print("Test Accuracy:", test_accuracy)
    print("Test Classification Report:\n", test_report)
    print("Test Confusion Matrix:\n", test_confusion)

    return None

if __name__ == "__main__":
    main()