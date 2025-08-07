import pandas as pd
from typing import List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

    
def get_dummies_cols(X_train, X_test, cols:List[str]):
    train_dummies = pd.get_dummies(X_train[cols], drop_first=True)
    test_dummies = pd.get_dummies(X_test[cols], drop_first=True)

    # Ensure the test set has the same dummy variables as the training set
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)
    
    X_train = pd.concat([X_train.drop(cols, axis=1), train_dummies], axis=1)
    X_test = pd.concat([X_test.drop(cols, axis=1), test_dummies], axis=1)

    return X_train, X_test

def evaluate_model(y_pred, y_test):
    """
    Evaluate a classifier and display:
    - Precision, Recall, Accuracy, and AUC
    """
    # # Get predictions and probabilities
    # y_pred = model.predict(X_test)
    # y_proba = model.predict_proba(X_test)[:, 1]  # Probability of the positive class

    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    # auc = roc_auc_score(y_test, y_proba)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    # print(f"AUC: {auc:.4f}")

    benchmark_precision = ((pd.Series(y_test).value_counts() * 100.0 / len(y_test)).round(1)[1]) / 100
    # test set label distribution
    
    print(f"Benchmark Precision: {benchmark_precision}") 
    print(f"Model's precision improvement: {(precision - benchmark_precision).round(4)}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        # "auc": auc
    }