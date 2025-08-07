from similarity_search.HNSW import HNSW
from cl_utils.cl_deep_network import ContrastivePolicyNetwork
import numpy as np
import pandas as pd
from src.utils import get_dummies_cols
from typing import List
import torch
from tqdm import tqdm

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
        
        self.df_train.drop(columns=['year', 'is_company_italian', 'target'], inplace=True)
        del self.df_test

        # variables for the HNSW index
        self.max_elements = len(data)
        self.embeddings_to_insert = None


    def init_model(self, 
                   model_path: str,
                   model_hyperparams: dict):
        
        input_dim = model_hyperparams['input_dim']
        hidden_dims = model_hyperparams['hidden_dims']
        embedding_dim = model_hyperparams['embedding_dim']
        dropout_rate = model_hyperparams['dropout_rate']
        temperature = model_hyperparams['temperature'] 
        self.embeddings_dim = embedding_dim

        if self.df_train.shape[1] != input_dim:
            raise ValueError(f"Input dimension mismatch: expected {input_dim}, got {self.df_train.shape[1]}")
        
        self.model = ContrastivePolicyNetwork(input_dim=input_dim,
                                              hidden_dims=hidden_dims,
                                              embedding_dim=embedding_dim,
                                              dropout_rate=dropout_rate,
                                              temperature=temperature)
        self.model.load_state_dict(torch.load(model_path))
        
    def embeddings(self, 
                   policies):
        """Generate embeddings for the given policies."""
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.get_embeddings(policies)
            
        embeddings = embeddings.cpu().numpy()
        return embeddings

    def batch_generate_embeddings(self):
        """Generate embeddings for the training set in batches."""
        all_embeddings = [] 
        with torch.no_grad():
            for i in tqdm(range(0, len(self.df_train), self.batch_size)):
                limit = min(i + self.batch_size, len(self.df_train))
                batch = [self.df_train.iloc[j].values.astype(np.float32) for j in range(i, limit)]
                tensors = torch.stack([torch.from_numpy(i) for i in batch])

                # Generate embeddings for the batch
                embeddings = self.embeddings(tensors)
                # print(embeddings.shape)
                all_embeddings.append(embeddings)

        self.embeddings_to_insert = np.vstack(all_embeddings)
        # print(self.embeddings_to_insert.shape)
    

    def build_and_insert_index(self,  
                               M: int, 
                               ef: int,
                               ef_construction: int, 
                               index_path: str):

        index = HNSW(embedding_size=self.embeddings_dim,
                     M=M,
                     ef=ef,
                     ef_construction=ef_construction,
                     index_path=index_path)

        index.build_index(max_elements=self.max_elements)
        index.add_embeddings(self.embeddings_to_insert, self.df_train)
        index.save_index() 





