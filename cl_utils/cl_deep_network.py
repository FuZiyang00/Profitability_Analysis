import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List

class PolicyEncoder(nn.Module):
    """
    Neural network encoder to transform policy features into embeddings.
    """

    def __init__(self, input_dim: int, 
                 hidden_dims: List[int] = [128, 64],
                 embedding_dim: int = 32,
                 dropout_rate: float = 0.2):
        
        super(PolicyEncoder, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        # Final embedding layer
        layers.append(nn.Linear(current_dim, embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: Output embeddings of shape (batch_size, embedding_dim).
        """
        return self.encoder(x)
    

class ContrastivePolicyNetwork(nn.Module):
    """
    Contrastive learning network for insurance policy classification.
    
    Architecture:
        1. Encodes query and candidate policies into embeddings
        2. Computes cosine similarities between query and candidates
        3. Applies softmax and cross-entropy loss
    """ 

    def __init__(self, input_dim: int,
                 hidden_dims: List[int] = [128, 64],
                 embedding_dim: int = 32,
                 dropout_rate: float = 0.2,
                 temperature: float = 0.1):

        super(ContrastivePolicyNetwork, self).__init__()

        self.encoder = PolicyEncoder(input_dim = input_dim,
                                     hidden_dims = hidden_dims, 
                                     embedding_dim = embedding_dim,
                                     dropout_rate = dropout_rate)

        self.temperature = temperature
        self.embedding_dim = embedding_dim

    def forward(self, query: torch.Tensor, candidates: torch.Tensor):
        """
        Forward pass of the contrastive network.
        
        Args:
            query: Query policies tensor of shape (batch_size, input_dim)
            candidates: Candidate policies tensor of shape (batch_size, n_candidates, input_dim)
        
        Returns:
            Similarity scores of shape (batch_size, n_candidates)
        """
        batch_size, n_candidates, input_dim = candidates.shape

        # Encode query policies
        query_embeddings = self.encoder(query)  # (batch_size, embedding_dim)

        # encode candidate policies
        flatten_candidates = candidates.view(batch_size * n_candidates, input_dim) 
        candidates_embeddings_flat = self.encoder(flatten_candidates)
        candidates_embeddings = candidates_embeddings_flat.view(batch_size, n_candidates, 
                                                           self.embedding_dim)

        # computing cosine similarity 
        query_embeddings_norm = F.normalize(query_embeddings, p=2, dim=1)
        candidates_embeddings_norm = F.normalize(candidates_embeddings, p=2, dim=1)

        similarities = torch.bmm(query_embeddings_norm.unsqueeze(1), 
                                candidates_embeddings_norm.transpose(1, 2)).squeeze(1)

        return similarities 
    
    def compute_loss(self, similarities: torch.Tensor, labels: torch.Tensor):
        """
        Contrastive loss 
        Args:
        similarities: Similarity scores of shape (batch_size, n_candidates)
        labels: True labels of shape (batch_size,)
        return:
        Cross-entropy loss 
        """

        loss = F.cross_entropy(similarities, labels)

        # Get predictions (highest similarity indices)
        predictions = torch.argmax(similarities, dim=1)  

        print(f"\nBatch Analysis (Loss: {loss.item():.4f}):")
        correct_count = 0
        for i in range(similarities.size(0)):
            pred_idx = predictions[i].item()
            true_idx = labels[i].item()

            if pred_idx == true_idx:
                correct_count += 1
        
        if i == similarities.size(0) - 1:
            print(f"{correct_count} correct out of {similarities.size(0)} samples")

        return loss
  
    def get_embeddings(self, policies: torch.Tensor): 

        if policies.dim() == 1:
            return self.encoder(policies.unsqueeze(0)).squeeze(0)
        else:
            return self.encoder(policies)
        

class ContrastivePolicyTrainer:
    """ Trainer class for the contrastivi policy network """

    def __init__(self, model: ContrastivePolicyNetwork, 
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4):
        
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), 
                                          lr=learning_rate, 
                                          weight_decay=weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                    mode='min', 
                                                                    patience=5, 
                                                                    factor=0.5)
        
    def train_epoch(self, dataloader: DataLoader, device: torch.device):
        """
        Function for training for one epoch
        """ 
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            query = batch['anchors'].to(device)
            candidates = batch['candidates'].to(device)
            labels = batch['labels'].to(device)

            # forward pass
            cosine_sim = self.model.forward(query, candidates)
            loss = self.model.compute_loss(cosine_sim, labels)

            # backward pass 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches +=1
            
        avg_loss = total_loss / num_batches
        return avg_loss 

    def evaluate(self, dataloader: DataLoader, device: torch.device):
        """
        Function for evaluating the model on a validation set
        """

        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0 
        with torch.no_grad():
            for batch in dataloader:
                query = batch['anchors'].to(device)
                candidates = batch['candidates'].to(device)
                labels = batch['labels'].to(device)

                similarities = self.model.forward(query, candidates)
                loss = self.model.compute_loss(similarities, labels)

                # Calculate accuracy
                predictions = torch.argmax(similarities, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            accuracy = correct_predictions / total_predictions
            
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    @staticmethod
    def train_contrastive_model(model, trainer, train_dataloader, validation_dataloader,
                                 device, n_epochs=100):
        
        """
        Complete training loop for the contrastive policy network.
        
        Args:
            model: ContrastivePolicyNetwork instance
            trainer: ContrastivePolicyTrainer instance
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            device: torch.device
            n_epochs: Number of training epochs
        """
        # Move model to device
        model.to(device)
        
        # Training history
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_loss = float('inf')

        print(f"Starting training for {n_epochs} epochs...")
        print(f"Device: {torch.device}")

        for epoch in range(n_epochs):
            # training phase 
            train_loss = trainer.train_epoch(train_dataloader, device)

            # Validation phase
            val_metrics = trainer.evaluate(validation_dataloader, device)
            val_loss = val_metrics['loss']
            val_accuracy = val_metrics['accuracy']

            # Update learning rate scheduler
            trainer.scheduler.step(val_loss)
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Print progress
            current_lr = trainer.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f} | "
                f"LR: {current_lr:.2e}")
        
        return {'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'best_val_loss': best_val_loss}