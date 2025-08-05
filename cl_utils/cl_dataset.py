from torch.utils.data import Dataset
import random
import numpy as np
from typing import List, Dict
import pandas as pd
import torch

class ContrastivePolicyDataset(Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 target_column: str = 'target',
                 n_negative: int = 4):
        
        """
        Custom Dataset for contrastive learning
        Creates training samples with:
        - anchor: query policy
        - candidates: list of n_negative opposite class policies + 1 same class policy
        - label: index of the same class policy in candidates list
        """

        self.df = df.copy()
        self.target_column = target_column
        self.n_negative = n_negative
        self.n_candidates = n_negative + 1  # +1 for the positive sample 

        # separate positive and negative indices
        self.class_0_indices = self.df[self.df[target_column] == 0].index.tolist()
        self.class_1_indices = self.df[self.df[target_column] == 1].index.tolist()
        # mapping class indices for easy access
        self.class_indices = {0: self.class_0_indices, 1: self.class_1_indices}

        # split features and targets 
        features_columns = self.df.drop(columns=[self.target_column]).columns.tolist()
        self.features = self.df[features_columns].values.astype(np.float32)
        self.targets = self.df[self.target_column].values.astype(np.int64)

        print(f"Dataset initialized:")
        print(f"- Features: {len(features_columns)}")
        print(f"- Negative samples per query: {self.n_negative}")
    
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx) -> Dict[str, any]:
        """
        Generate a contrastive learning sample.
        
        Args:
            idx: Index of the anchor sample
            
        Returns:
            Dictionary containing:
            - anchor: numpy array of anchor policy features
            - candidates: list of numpy arrays (n_negative opposite + 1 same class)
            - label: index of the same class policy in candidates list
        """

        if idx >= len(self):
            raise IndexError("Index out of bounds for dataset length.")
        
        # get anchor sample
        anchor_features = self.features[idx]
        anchor_target = self.targets[idx]

        # determine opposite class
        opposite_class = 1 - anchor_target
        same_class = anchor_target

        # sample negative candidates
        negative_indices = self._sample_indices(self.class_indices[opposite_class],
                                                n_samples=self.n_negative,
                                                exclude_idx=idx)

        # sample one positive candidate from the same class
        positive_indices = self._sample_indices(self.class_indices[same_class],
                                                n_samples=1,
                                                exclude_idx=idx)
        
        # create candidates list
        candidates = []
        for neg_idx in negative_indices:
            candidates.append(self.features[neg_idx])
        candidates.append(self.features[positive_indices[0]])

        # shuffle the candidates
        shuffled_indices = list(range(len(candidates)))
        random.shuffle(shuffled_indices)

        # Reorder candidates according to shuffled indices
        shuffled_candidates = [candidates[i] for i in shuffled_indices]

        # Find the new position of the positive sample
        original_positive_idx = len(candidates) - 1  # Was last
        label = shuffled_indices.index(original_positive_idx)

        candidates = shuffled_candidates

        return {
            'anchors': anchor_features,
            'candidates': candidates,
            'labels': label
        }

    def _sample_indices(self, class_indices: List[int],
                        n_samples: int,
                        exclude_idx: int = None):

        # Remove exclude_idx if it exists in class_indices
        available_indices = [idx for idx in class_indices if idx != exclude_idx]

        if len(available_indices) < n_samples:
            # sample with replacement if not enough samples
            sample_indices = np.random.choice(available_indices,
                                              size = n_samples,
                                              replace=True).tolist()

        else:
            # sample without replacement
            sample_indices = np.random.choice(available_indices,
                                              size = n_samples,
                                              replace=False).tolist()

        return sample_indices

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle the contrastive learning samples.
        """
        anchors = torch.stack([torch.from_numpy(item['anchors']) for item in batch])
        labels = torch.tensor([item['labels'] for item in batch])

        # Keep candidates as nested list structure for flexible processing
        candidates_batch = []
        for item in batch:
            item_candidates = [torch.from_numpy(c) for c in item['candidates']]
            item_candidates_stacked = torch.stack(item_candidates)
            # Shape: (n_candidates, input_dim)
            candidates_batch.append(item_candidates_stacked)

        candidates = torch.stack(candidates_batch)

        return {'anchors': anchors,
                'candidates': candidates,  # List of lists of tensors
                'labels': labels}


