import faiss
import os
import numpy as np
import pandas as pd

class HNSW:

    def __init__(self, embedding_size: int,
                 M: int, # number of bi-directional links created for each element
                         # size of thea adjacency list for each node

                 ef: int, # exploration factor: how many candidates are evaluated during a search
                          # how many canidate nodes are considered in the greedy search

                 ef_construction: int, # how many existing vectors are considered as potential neighbors
                 index_path: str):
        
        self.embedding_size = embedding_size
        self.M = M
        self.ef = ef
        self.ef_construction = ef_construction
        self.index_path = index_path
        self.index = None

    def build_index(self, max_elements: int):
        self.index = faiss.IndexHNSWFlat(self.embedding_size, self.M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.maxElements = max_elements
    
    def add_embeddings(self, embeddings: np.ndarray, df: pd.DataFrame):
        if self.index is None:
            raise ValueError("Index not initialized. Call build_index first.")
        
        if embeddings.shape[1] != self.embedding_size:
            raise ValueError(f"Embeddings must have shape (n, {self.embedding_size})")

        if embeddings.shape[0] > self.index.hnsw.maxElements:
            raise ValueError(f"Number of embeddings exceeds max_elements: {self.index.hnsw.maxElements}")
        
        self.index.add_with_ids(embeddings, np.array(df.index).astype(np.int64))
    
    def search(self, query, k: int = 5):
        """
        Search for the k nearest neighbors of the query vector.
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_hnsw_index first.")
        
        self.index.hnsw.efSearch = self.ef
        distances, indices = self.index.search(query, k)
        
        return [index for index in indices[0]] 
    
    def save_index(self):
        if self.index is None:
            raise ValueError("Index not built. Call build_hnsw_index first.")
        
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)
        
        faiss.write_index(self.index, os.path.join(self.index_path, 'hnsw_index.faiss'))