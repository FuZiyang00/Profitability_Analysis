from sklearn.model_selection import train_test_split
from typing import List
import pandas as pd
from xgboost.callback import EarlyStopping
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_VALIDATION_SET_SIZE = 0.20
MODEL_VALIDATION_SPLIT_RANDOM_STATE = 42
N_ESTIMATORS = 500
EARLY_STOPPING_ROUNDS = 50

class XG_Boost_Model:

    def __init__(self, data: pd.DataFrame, target: str, params: dict):
        self.data = data
        self.target = target
        self.model = None
        self.params = params

    
    # functions to split train and test 
    def oot_train_test_split(self, features_to_drop:List[str] = None, oot=False):

        if features_to_drop: 
            self.data.drop(features_to_drop, axis=1, inplace=True)

        if oot:
            split_column = 'year'
            train = self.data[self.data[split_column] < oot]
            test = self.data[self.data[split_column] >= oot]

            # splitting the data into X and y
            X_train = train.drop([self.target], axis=1)
            y_train = train[self.target]
            X_test = test.drop([self.target], axis=1)
            y_test = test[self.target]
        
        else: 
            X = self.data.drop([self.target], axis=1)
            y = self.data[self.target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                test_size=0.25,
                                                                stratify=y,
                                                                random_state=42)

        print(f"Training vs Validation size: {X_train.shape[0]} - {X_test.shape[0]}")
        print("Training set Class distribution:")
        print((y_train.value_counts()*100.0/len(y_train)).round(1))
        print("Test set Class distribution:")
        print((y_test.value_counts()*100.0/len(y_test)).round(1))
    
        return X_train, X_test, y_train, y_test


    @staticmethod
    def get_dummies_cols(X_train, X_test, cols:List[str]):
        train_dummies = pd.get_dummies(X_train[cols], drop_first=True)
        test_dummies = pd.get_dummies(X_test[cols], drop_first=True)

        # Ensure the test set has the same dummy variables as the training set
        test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)
        
        X_train = pd.concat([X_train.drop(cols, axis=1), train_dummies], axis=1)
        X_test = pd.concat([X_test.drop(cols, axis=1), test_dummies], axis=1)

        return X_train, X_test
    

    def xgb_fit_model(self, X_train, y_train):
        """Fits a model of XGB with given features and targets."""
        X_full = X_train.copy()
        y_full = y_train.copy()

        # Split the data into training and validation sets
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train,
            y_train,
            test_size=MODEL_VALIDATION_SET_SIZE,
            random_state=MODEL_VALIDATION_SPLIT_RANDOM_STATE,
            stratify=y_train,
        )

        # FIRST MODEL #
        # Parametrize a watch list to follow the training performance
        # watch_list = [(X_train, y_train), (X_validation, y_validation)]

        # # Configure early stopping
        # early_stop = EarlyStopping(
        #     rounds=max(10, int(0.1 * len(X_train))),   # Minimum patience (stop if no improvement for 10 rounds)
        #     metric_name='aucpr',    # Metric to monitor (e.g., 'error', 'auc', 'merror')
        #     data_name='validation_0', # Name of the eval_set (default for first set)
        #     save_best=True            # Keeps the best model (not just the last one)
        # )

        # Fit the model
        model = XGBClassifier(
            **self.params,
            scale_pos_weight=(
                y_train.value_counts().sort_index()[0]
                / y_train.value_counts().sort_index()[1]
                ),
            n_estimators=N_ESTIMATORS,  # Set high; early stopping will trim it
            early_stopping_rounds=EARLY_STOPPING_ROUNDS
        )

        # Fitting first model with 2 eval set elements which the first one is training itself.
        first_model = model.fit(X_train, 
                                y_train, 
                                eval_set=[(X_validation, y_validation)], 
                                verbose=100)

        # Retrieve the best iteration
        self.params["n_estimators"] = first_model.best_iteration
        # print("Working with parameters:", self.params)
        print(f"Best iteration: {first_model.best_iteration}")
        print(f"Best score: {first_model.best_score}")

        # Fit the final model with the best number of estimators
        final_model = XGBClassifier(**self.params,
                                    scale_pos_weight=(y_full.value_counts().sort_index()[0] / y_full.value_counts().sort_index()[1]),)
        
        self.model = final_model.fit(X_full, y_full)
        print("Done !")


    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """
        Evaluate a classifier and display:
        - Precision, Recall, Accuracy, and AUC
        """
        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # Probability of the positive class

        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC: {auc:.4f}")

        benchmark_precision = ((y_test.value_counts()*100.0/len(y_test)).round(1)[1])/100 # test set label distribution
        print(f"Benchmark Precision: {benchmark_precision}") 
        print(f"Model's precision improvement: {(precision - benchmark_precision).round(4)}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc
        }