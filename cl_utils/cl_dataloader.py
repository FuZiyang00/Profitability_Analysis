import torch
from torch.utils.data import DataLoader, Dataset

class ContrastivePOLICYDataLoader(DataLoader):
    def __init__(self, 
                 dataset: Dataset, 
                 batch_size: int = 32):
        """
        Custom DataLoader for contrastive learning datasets.
        
        Args:
            dataset (Dataset): The dataset to load.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data at every epoch.
            num_workers (int): Number of subprocesses to use for data loading.
        """
        super().__init__(dataset, batch_size=batch_size, shuffle=True) 
        
        self.dataloader = DataLoader(dataset=dataset, 
                                     batch_size=batch_size, 
                                     shuffle=True,
                                     collate_fn=self.collate_fn)

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

    def print_info(self):
        """
        Print dataset information.
        """
        # Test shuffling by checking label distribution
        batch = next(iter(self.dataloader))
        print(f"Batch shapes:")
        print(f"- Anchors: {batch['anchors'].shape}")
        print(f"- Candidates: {batch['candidates'].shape}")
        print(f"- Labels: {batch['labels'].shape}")
        print(f"- Labels values: {batch['labels'].tolist()}")


