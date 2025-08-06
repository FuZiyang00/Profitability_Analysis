from cl_utils.cl_deep_network import ContrastivePolicyTrainer, ContrastivePolicyNetwork
from cl_utils.cl_dataset import ContrastivePolicyDataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
from src.utils import get_dummies_cols
from typing import List
 
class ContrastiveLearningPipeline:
    """Pipeline for contrastive learning with a custom dataset and deep network."""

    def __init__(self, data: pd.DataFrame, 
                 train_oot: int = 2022,
                 val_oot: int = 2020,
                 dummy_cols: List[str] = None): 
        
        train = data[data["year"] < train_oot].reset_index(drop=True)
        test = data[data["year"] >= train_oot].reset_index(drop=True)

        # split train and test sets
        train_df, test_df = get_dummies_cols(train, test, dummy_cols) 
        test_df.drop(columns=['year', 'is_company_italian'], inplace=True)
        self.test_df = test_df

        # split train_df into train and validation sets
        val_df = train_df[train_df['year'] > val_oot].reset_index(drop=True)
        training_df = train_df[train_df['year'] <= val_oot].reset_index(drop=True)
        training_df.drop(columns=['year', 'is_company_italian'], inplace=True) 
        val_df.drop(columns=['year', 'is_company_italian'], inplace=True)
        self.val_df = val_df 
        self.training_df = training_df

        self.train_dataset = ContrastivePolicyDataset(training_df)
        self.val_dataset = ContrastivePolicyDataset(val_df)

        self.training_dataloader = None
        self.val_dataloader = None
        self.model = None 
        self.trainer = None

    def dataloaders(self, 
                    batch_size: int = 16): 
        
        """Create DataLoader objects for training and validation datasets."""
        self.training_dataloader = DataLoader(self.train_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=True, 
                                              drop_last=True,
                                              collate_fn=ContrastivePolicyDataset.collate_fn)
        
        batch = next(iter(self.training_dataloader))
        print(f"Batch shapes:")
        print(f"- Anchors: {batch['anchors'].shape}")
        print(f"- Candidates: {batch['candidates'].shape}")
        print(f"- Labels: {batch['labels'].shape}")

        self.val_dataloader = DataLoader(self.val_dataset, 
                                         batch_size=batch_size, 
                                         shuffle=True, 
                                         drop_last=True,
                                         collate_fn=ContrastivePolicyDataset.collate_fn)

        batch = next(iter(self.val_dataloader))
        print(f"Batch shapes:")
        print(f"- Anchors: {batch['anchors'].shape}")
        print(f"- Candidates: {batch['candidates'].shape}")
        print(f"- Labels: {batch['labels'].shape}")

    
    def init_model(self, 
                   hidden_dims=[256, 128], 
                   embedding_dim=64,
                   dropout_rate=0.1,
                   temperature=0.07): 
        
        """Initialize the contrastive learning model."""
        input_dim = self.training_df.shape[1] - 1  # Exclude target column
        self.model = ContrastivePolicyNetwork(input_dim=input_dim,
                                               hidden_dims=hidden_dims,
                                               embedding_dim=embedding_dim,
                                               dropout_rate=dropout_rate,
                                               temperature=temperature)
        
        return {"input_dim": input_dim,
                "hidden_dims": hidden_dims,
                "embedding_dim": embedding_dim,
                "dropout_rate": dropout_rate,
                "temperature": temperature}
    
    def init_trainer(self,
                     learning_rate: float= 5e-4,
                     weight_decay: float = 1e-5): 
        
        """Initialize the trainer for the contrastive learning model."""
        self.trainer = ContrastivePolicyTrainer(model = self.model,
                                                learning_rate = learning_rate,      
                                                weight_decay = weight_decay)
    
    def train(self, 
              device: str,
              epochs: int = 100,
              save_path: str = None):
        
        """Train the contrastive learning model."""
        training_metrics = ContrastivePolicyTrainer.train_contrastive_model(self.model, 
                                                                            self.trainer, 
                                                                            self.training_dataloader,
                                                                            self.val_dataloader,
                                                                            device,
                                                                            epochs)
        torch.save(self.model.state_dict(), save_path)
        return training_metrics