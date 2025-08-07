from similarity_search.HNSW import HNSW
import torch
import faiss
from tqdm import tqdm
import numpy as np
import pandas as pd
from src.utils import get_dummies_cols, evaluate_model
from typing import List
from cl_utils.cl_deep_network import ContrastivePolicyNetwork
import statistics

class EvaluateCLModel:
    """Pipeline for evaluating a Contrastive Learning model using HNSW index."""

    def __init__(self, 
                 data: pd.DataFrame, 
                 dummy_cols: List[str], 
                 index_path: str): 
        
        self.df_train = data[data["year"] < 2022].reset_index(drop=True)
        self.df_test = data[data["year"] >= 2022].reset_index(drop=True)
    
        self.df_train, self.df_test = get_dummies_cols(self.df_train, 
                                                       self.df_test, 
                                                       dummy_cols) 
        
        self.features_columns = self.df_test.drop(columns=['year', 
                                                      'is_company_italian', 
                                                      'target']).columns.tolist()
        
        self.test_features = self.df_test[self.features_columns].values.astype(np.float32)
        self.test_targets = self.df_test["target"].values.astype(np.int64)

        self.index = faiss.read_index(index_path)
        self.model = None


    def init_model(self, 
                   model_path: str,
                   model_hyperparams: dict):
        
        input_dim = model_hyperparams['input_dim']
        hidden_dims = model_hyperparams['hidden_dims']
        embedding_dim = model_hyperparams['embedding_dim']
        dropout_rate = model_hyperparams['dropout_rate']
        temperature = model_hyperparams['temperature'] 
        self.embeddings_dim = embedding_dim

        if len(self.features_columns) != input_dim:
            raise ValueError(f"Input dimension mismatch: expected {input_dim}, got {len(self.features_columns)}")
        
        self.model = ContrastivePolicyNetwork(input_dim=input_dim,
                                              hidden_dims=hidden_dims,
                                              embedding_dim=embedding_dim,
                                              dropout_rate=dropout_rate,
                                              temperature=temperature)
        self.model.load_state_dict(torch.load(model_path))     

    
    def get_labels(self, 
                   indexes: List[int]) -> List[int]:
        
        """Get labels for the given indexes."""
        predicted_labels = []
        for idx in indexes:
            value = self.df_train.at[idx, "target"]
            predicted_labels.append(value)
            
        return statistics.mode(predicted_labels) 


    def evaluate(self, k: int = 5):

        """Evaluate the model using the HNSW index."""
        if self.model is None:
            raise ValueError("Model not initialized. Call init_model first.")
        
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for i in tqdm(range(0, len(self.test_features), 1)):
                policy = self.test_features[i]
                policy_tensor = torch.from_numpy(policy)
                policy_embedding = self.model.get_embeddings(policy_tensor)
                policy_vector = policy_embedding.cpu().numpy()

                # Search for the k nearest neighbors
                indices = HNSW.search(self.index, policy_vector, k=k)
                
                predicted_label = self.get_labels(indices)
                y_pred.append(predicted_label)
        
        y_pred = np.array(y_pred).astype(np.int64)

        if len(y_pred) != len(self.test_targets):
            raise ValueError(f"Length of predictions {len(y_pred)} does not match length of test targets {len(self.test_targets)}")
        
        metrics = evaluate_model(y_pred, self.test_targets)

        return metrics



     

        