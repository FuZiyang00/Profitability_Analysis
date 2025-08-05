from xgb_utils.xgb_pipeline import XGB_Pipeline
import pandas as pd

if __name__ == "__main__":

    data = pd.read_csv('data/final_data.csv', sep=';', low_memory=False)  
    target = 'target'  
    hyper_params = {'eta': 0.01, 
                    'gamma': 0.1, 
                    'max_depth': 6, 
                    'subsample': 0.2, 
                    'colsample_bytree': 1, 
                    'objective': 'binary:logistic', 
                    'base_score': 0.5, 
                    'eval_metric': 'aucpr', 
                    'seed': 42, 
                    'min_child_weight': 2, 
                    'reg_alpha': 2, 
                    'importance_type': 'gain'}
    
    xgb_pipeline = XGB_Pipeline(data, target, hyper_params, oot=2022)
    dummy_cols = ['Sector', "REGION_GROUP"]
    performance_metrics = xgb_pipeline.run_pipeline(dummy_cols = dummy_cols)  
