from xgb_utils.xgb_model import XG_Boost_Model 
from typing import List
import pandas as pd
from src.utils import get_dummies_cols, evaluate_model

class XGB_Pipeline():
    """Pipeline for XGBoost model training and evaluation."""

    def __init__(self, data: pd.DataFrame, target: str, 
                 hyper_params: dict, oot: int = 2022):

        self.hyper_params = hyper_params
        self.xgb = XG_Boost_Model(data, target, hyper_params)
        self.X_train, self.X_test, self.y_train, self.y_test = self.xgb.oot_train_test_split(oot=oot)
        self.model = None
        self.X_train_final, self.X_test_final = None, None
    
    def training(self, dummy_cols: List[str] = None):
        """
        Train the XGBoost model with the provided training data.
        """
        self.X_train_final, self.X_test_final = get_dummies_cols(self.X_train, 
                                                                 self.X_test, 
                                                                 cols = dummy_cols)
        
        self.X_train_final.drop(columns=['year', 'is_company_italian'], inplace=True) 
        self.X_test_final.drop(columns=['year', 'is_company_italian'], inplace=True)

        print(self.X_train_final.shape, self.X_test_final.shape) 
        print("--" * 30)
        print("Training XGBoost model")
        
        self.xgb.xgb_fit_model(self.X_train_final, self.y_train)
        self.model = self.xgb.model
    
    def evaluation(self):
        """
        Evaluate the trained model on the test set.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call training() first.")
        
        print("--" * 30)
        print("Evaluating model performance on the test set")

        metrics = evaluate_model(self.model, self.X_test_final, self.y_test)
        return metrics
    
    def run_pipeline(self, dummy_cols: List[str] = None):
        """
        Run the entire pipeline: training and evaluation.
        """
        self.training(dummy_cols)
        metrics = self.evaluation()
        return metrics
