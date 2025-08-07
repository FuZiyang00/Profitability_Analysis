from similarity_search.evaluate_model import EvaluateCLModel
import pandas as pd
import json
import os
from src.utils import evaluate_model
import xgboost as xgb


if __name__ == "__main__":
    
    data = pd.read_csv('data/final_data.csv', sep=';', low_memory=False)

    if os.path.exists('models/contrastive_model.pth') and os.path.exists('models/xgb_model.json'):
        with open('models/contrastive_model_hyperparams.json', 'r') as file:
            hyper_params = json.load(file)

        index_path = 'models/hnsw_index/hnsw_index.faiss'

        evaluator = EvaluateCLModel(data=data,
                                    dummy_cols=['Sector', "REGION_GROUP"],
                                    index_path=index_path)
        
        evaluator.init_model(model_path='models/contrastive_model.pth',
                             model_hyperparams=hyper_params)
        
        cl_model_metrics = evaluator.evaluate()

        xgboost_model = xgb.XGBClassifier()
        xgboost_model.load_model('models/xgb_model.json')
        y_pred = xgboost_model.predict(evaluator.test_features)
        xgboost_metrics = evaluate_model(y_pred, evaluator.test_targets)
        print()
        print("Evaluation Results:")
        print()
        print("|Model|Accuracy|Precision|Recall|")
        print("|-----|--------|---------|------|")
        print(f"|CL   |{cl_model_metrics['accuracy']:.4f}  |{cl_model_metrics['precision']:.4f}   |{cl_model_metrics['recall']:.4f}|")
        print("|-----|--------|---------|------|")
        print(f"|XGB  |{xgboost_metrics['accuracy']:.4f}  |{xgboost_metrics['precision']:.4f}   |{xgboost_metrics['recall']:.4f}|")



    else:
        print("Contrastive Learning model not found. Please train the model first.")
        exit(1)
