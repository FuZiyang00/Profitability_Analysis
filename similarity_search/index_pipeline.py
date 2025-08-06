from similarity_search.HNSW import HNSW
from cl_utils.cl_deep_network import ContrastivePolicyNetwork
import numpy as np
import pandas as pd
from src.utils import get_dummies_cols
from typing import List
import torch

class HNSWPipeline:
    """Pipeline for building and using an HNSW index for similarity search."""

    #1. Initialize the Model 
    #2. Convert Observations format suitable for the Model
    #3. Build the HNSW index
    #4. Save the index to disk 

    def __init__(self, data: pd.DataFrame, 
                 batch_size: int, 
                 dummy_cols: List[str]): 
        
        self.df_train = data[data["year"] < 2022].reset_index(drop=True)
        self.df_test = data[data["year"] >= 2022].reset_index(drop=True)
        self.batch_size = batch_size
    
        self.df_train, self.df_test = get_dummies_cols(self.df_train, 
                                                       self.df_test, 
                                                       dummy_cols) 
        
        self.df_train.drop(columns=['year', 'is_company_italian'], inplace=True)
        self.df_test.drop(columns=['year', 'is_company_italian'], inplace=True)
    
    def embeddings(self, 
                   policies,
                   model_path: str, 
                   model_hyperparams: dict):

        input_dim = model_hyperparams['input_dim']
        hidden_dims = model_hyperparams['hidden_dims']
        embedding_dim = model_hyperparams['embedding_dim']
        dropout_rate = model_hyperparams['dropout_rate']
        temperature = model_hyperparams['temperature'] 

        self.model = ContrastivePolicyNetwork(input_dim=input_dim,
                                              hidden_dims=hidden_dims,
                                              embedding_dim=embedding_dim,
                                              dropout_rate=dropout_rate,
                                              temperature=temperature)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        with torch.no_grad():
            embeddings = self.model.get_embeddings(policies)
            
        embeddings = embeddings.cpu().numpy()
        return embeddings





