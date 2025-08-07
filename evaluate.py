from similarity_search.evaluate_model import EvaluateCLModel
import pandas as pd
import json
import os
import torch


if __name__ == "__main__":
    
    data = pd.read_csv('data/final_data.csv', sep=';', low_memory=False)

    if os.path.exists('models/contrastive_model.pth'):
        with open('models/contrastive_model_hyperparams.json', 'r') as file:
            hyper_params = json.load(file)

        index_path = 'models/hnsw_index/hnsw_index.faiss'

        evaluator = EvaluateCLModel(data=data,
                                    dummy_cols=['Sector', "REGION_GROUP"],
                                    index_path=index_path)
        
        evaluator.init_model(model_path='models/contrastive_model.pth',
                             model_hyperparams=hyper_params)
        
        evaluator.evaluate()

    else:
        print("Contrastive Learning model not found. Please train the model first.")
        exit(1)
