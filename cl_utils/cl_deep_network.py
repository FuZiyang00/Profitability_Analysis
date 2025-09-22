import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List
import pandas as pd

class ContrastivePolicyNetwork(nn.Module):
    """
    Complete contrastive learning network for policy representations.
    Combines encoder and contrastive learning in a single class.
    """
    def __init__(self, input_dim: int, 
                 hidden_dims: List[int], 
                 embedding_dim: int,
                 dropout: float = 0.1, 
                 temperature: float = 0.07):
        super(ContrastivePolicyNetwork, self).__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        
        # Final embedding layer
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.encoder = nn.Sequential(*layers)
        self.temperature = temperature

    def encoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode policies to normalized embeddings.
        
        Args:
            x: Input policy representation [batch_size, input_dim]
            
        Returns:
            embeddings: L2-normalized policy embeddings [batch_size, embedding_dim]
        """
        embeddings = self.encoder(x)
        return F.normalize(embeddings, p=2, dim=1)
    
    def forward(self, 
                training_policy: torch.Tensor,
                similar_policy: torch.Tensor,
                dissimilar_policies: torch.Tensor):
        
        """
        Forward pass for contrastive learning with efficient batched encoding.
        
        Args:
            training_policy: [batch_size, input_dim]
            similar_policy: [batch_size, input_dim]
            dissimilar_policies: [batch_size, num_dissimilar, input_dim]
            
        Returns:
            loss: Contrastive loss
            similarities: Similarity scores for monitoring
        """

        batch_size, num_dissimilar, input_dim = dissimilar_policies.shape

        training_input = torch.cat([
            training_policy, # [batch_size, input_dim]
            similar_policy,   # [batch_size, input_dim]
            dissimilar_policies.view(batch_size * num_dissimilar, input_dim)# [batch_size * num_dissimilar, input_dim]
        ], dim=0)

        all_embeddings = self.encoder_forward(training_input)

        # Split embeddings
        training_emb = all_embeddings[:batch_size]  # [batch_size, embedding_dim]
        similar_emb = all_embeddings[batch_size:2*batch_size]  # [batch
        dissimilar_embs = all_embeddings[2*batch_size:].view(batch_size, num_dissimilar, -1)  # [batch_size, num_dissimilar, embedding_dim]

        # Compute similarities
        # Similarity with similar policy (positive pairs)
        sim_positive = torch.sum(training_emb * similar_emb, dim=1) / self.temperature  # [batch_size]
        
        # Similarities with dissimilar policies (negative pairs)
        sim_negative = torch.bmm(dissimilar_embs, training_emb.unsqueeze(2)).squeeze(2) / self.temperature  # [batch_size, num_dissimilar]
        
        # Combine similarities
        all_similarities = torch.cat([sim_positive.unsqueeze(1), sim_negative], dim=1)  # [batch_size, 1 + num_dissimilar]
        
        return all_similarities
    
    def compute_loss(self, similarities: torch.Tensor, 
                     labels: torch.Tensor) -> torch.Tensor:
        """
        Contrastive loss 
        Args:
            similarities: Similarity scores of shape (batch_size, n_candidates)
            labels: True labels of shape (batch_size,)
        Returns:
            Cross-entropy loss 
        """

        return F.cross_entropy(similarities, labels)
    

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
        correct_predictions = 0
        total_predictions = 0
        num_batches = 0

        for batch in dataloader:
            query = batch['anchors'].to(device)
            sim_pol = batch['sim_pol'].to(device)
            candidates = batch['candidates'].to(device)
            labels = batch['labels'].to(device)

            # forward pass
            similarities = self.model.forward(query, sim_pol, candidates)
            loss = self.model.compute_loss(similarities, labels)

            # Calculate accuracy
            predictions = torch.argmax(similarities, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            # backward pass 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_accuracy = correct_predictions / total_predictions

        return {'avg_loss': avg_loss,
                'avg_accuracy': avg_accuracy}

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
                sim_pol = batch['sim_pol'].to(device)
                candidates = batch['candidates'].to(device)
                labels = batch['labels'].to(device)

                similarities = self.model.forward(query, sim_pol, candidates)
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
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        epochs = []

        print(f"Starting training for {n_epochs} epochs...")
        print(f"Device: {torch.device}")

        for epoch in range(n_epochs):
            
            # training step
            train_metrics = trainer.train_epoch(train_dataloader, device)
            train_loss = train_metrics['avg_loss']
            train_accuracy = train_metrics['avg_accuracy']

            # validation step
            val_metrics = trainer.evaluate(validation_dataloader, device)
            val_loss = val_metrics['loss']
            val_accuracy = val_metrics['accuracy']

            # Update learning rate scheduler
            trainer.scheduler.step(val_loss)
            
            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            epochs.append(epoch + 1)

            # Print progress
            current_lr = trainer.optimizer.param_groups[0]['lr']

            print(f"\nEpoch {epoch+1:3d} Analysis:")
            print("T_Acc | T_Loss | V_Acc | V_Loss   | LR      |")
            print("------|--------|-------|----------|----------")
            print(f" {train_accuracy:.2f} | {train_loss:.2f}   | {val_accuracy:.2f}  | {val_loss:.2f}     | {current_lr:.2e} |")

        metrics = {'epoch': epochs,
                   'train_losses': train_losses,
                   'train_accuracies': train_accuracies,
                   'val_losses': val_losses,
                   'val_accuracies': val_accuracies}
        
        df_metrics = pd.DataFrame(metrics)
        return df_metrics