from src.data_pipeline import DataProcessingPipeline
import pandas as pd

if __name__ == "__main__":
    # Example usage
    initial_data = pd.read_csv('data/initial_data.csv', sep=';', low_memory=False)  # Load your data here
    pipeline = DataProcessingPipeline(initial_data)
    pipeline.run_pipeline()

    final_data = pipeline.final_data
    
    