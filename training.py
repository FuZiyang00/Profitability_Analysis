from xgb_utils.xgb_pipeline import XGB_Pipeline
from cl_utils.cl_pipeline import ContrastiveLearningPipeline
import pandas as pd
import torch
import json

if __name__ == "__main__":
    
    data = pd.read_csv('data/final_data.csv', sep=';', low_memory=False)

    print("Training XGBoost model...")

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
    xgb_pipeline.run_pipeline(dummy_cols = dummy_cols, 
                              save_path='models/xgb_model.json')  

    print("--" * 30)

    print("Training Contrastive Learning model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    pipeline = ContrastiveLearningPipeline(data, dummy_cols = dummy_cols)

    # initialize dataloaders
    pipeline.dataloaders(batch_size=16)

    # initialize model
    hyper_params = pipeline.init_model()
    # write hyperparameters to a JSON file
    with open('models/contrastive_model_hyperparams.json', 'w') as file:
        json.dump(hyper_params, file, indent=4)

    print(f"Working with : {hyper_params}")

    # initialize trainer
    pipeline.init_trainer() 

    # train model
    training_metrics = pipeline.train(device, 
                                      save_path='models/contrastive_model.pth') 

