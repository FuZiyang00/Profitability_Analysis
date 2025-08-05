from cl_utils.cl_pipeline import ContrastiveLearningPipeline
import pandas as pd
import torch

if __name__ == "__main__":

    data = pd.read_csv('data/final_data.csv', sep=';', low_memory=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dummy_cols = ['Sector', "REGION_GROUP"]
    pipeline = ContrastiveLearningPipeline(data, dummy_cols = dummy_cols)

    # initialize dataloaders
    pipeline.dataloaders(batch_size=16)

    # initialize model
    pipeline.init_model()

    # initialize trainer
    pipeline.init_trainer() 

    # train model
    training_metrics = pipeline.train(device, 
                                      save_path='contrastive_model.pth') 
