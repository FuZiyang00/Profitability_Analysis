import pandas as pd 
import numpy as np 
from src.data_processing import Data_Imputation
from tqdm import tqdm

class DataProcessingPipeline:
    """
    Class to handle the data processing pipeline including imputation and feature engineering.
    """

    def __init__(self, initial_data: pd.DataFrame):
        self.initial_data = initial_data
        self.imputed_data = None
        self.final_data = None

    def DataImputation(self):
        """    
        Function to perform data imputation on the initial dataset.
        Args:
            initial_data (pd.DataFrame): The initial dataset to be processed.
        Returns:
            pd.DataFrame: The dataset after performing data imputation.
        """

        nans = Data_Imputation.percentage_missing_df(self.initial_data)
        full_empty_columns = nans[nans['Missing_Percentage'] == 100]['Column'].tolist()
        
        # Remove columns with 100% missing values
        self.initial_data.drop(columns=full_empty_columns, inplace=True, errors='ignore')

        del nans, full_empty_columns

        # We want to impute based on sector proximity, so if drop observations with missing sector
        self.initial_data.dropna(subset=['Sector'], inplace=True)

        # Split the data into training and test sets
        train_df = self.initial_data[self.initial_data['year'] < 2022].copy()
        test_df = self.initial_data[self.initial_data['year'] >= 2022].copy()
        del self.initial_data 

        # Impute missing turnover values
        nans = Data_Imputation.percentage_missing_df(train_df)
        missing_cols = nans['Column'].tolist()

        if "TURNOVER_M" in missing_cols:
            sector_medians = train_df.groupby('Sector')['TURNOVER_M'].median()
            
            # Impute missing values using sector medians
            train_df['TURNOVER_M'] = train_df.apply(
                lambda row: row['TURNOVER_M'] if pd.notna(row['TURNOVER_M']) 
                else sector_medians[row['Sector']], axis=1)
            
            test_df['TURNOVER_M'] = test_df['TURNOVER_M'].fillna(
                test_df['Sector'].map(sector_medians))

        # save a copy of the train_df with imputed values
        source_train_df = train_df.copy()

        print("Starting Imputation")
        for col in tqdm(missing_cols, desc='imputing items'):
            if col == 'REGION_GROUP':
                continue
            train_df[col] = train_df.apply(lambda row: Data_Imputation.turnover_proximity_imputer(row, 
                                                                                            source_train_df, col), axis=1)             

        for col in tqdm(missing_cols, desc='imputing items'):
            if col == 'REGION_GROUP':
                continue
            test_df[col] = test_df.apply(lambda row: Data_Imputation.turnover_proximity_imputer(row, 
                                                                                            source_train_df, col), axis=1)

        sector_region_mode = train_df.groupby('Sector')['REGION_GROUP'].agg(lambda x: x.mode()[0])
        
        # Impute missing REGION values using sector-wise mode
        train_df['REGION_GROUP'] = train_df['REGION_GROUP'].fillna(
            train_df['Sector'].map(sector_region_mode))

        test_df['REGION_GROUP'] = test_df['REGION_GROUP'].fillna(
            test_df['Sector'].map(sector_region_mode))

        self.imputed_data = pd.concat([train_df, test_df], axis=0)
        
        # drop columns that are still missing values
        nans_table = Data_Imputation.percentage_missing_df(self.imputed_data)
        print(len(nans_table), "columns still have missing values after imputation.")
        missing_cols = nans_table['Column'].tolist()
        drop_cols = [col for col in missing_cols if (col.endswith('_mean') or col.endswith('_median'))]
        self.imputed_data.dropna(subset=drop_cols, inplace=True)
        self.imputed_data = self.imputed_data.reset_index(drop=True)

        nans_table = Data_Imputation.percentage_missing_df(self.imputed_data)
        print(f'End of imputation, remaining missing columns: {len(nans_table)}')


    def FeatureEngineering(self):
        """
        Function to perform feature engineering on the imputed dataset.
        Args:
            imputed_data (pd.DataFrame): The dataset after performing data imputation.
        Returns:
            pd.DataFrame: The dataset after performing feature engineering.
        """
        print("Starting Feature Engineering")
        # sector-wise median calculations
        self.imputed_data['TURNOVER_SIZE'] = self.imputed_data["TURNOVER_M"] / self.imputed_data["TURNOVER_M_median"]
        self.imputed_data['PAYMENT_SPEED'] = self.imputed_data["CREDITOR_DY_Q"] / self.imputed_data["CREDITOR_DY_Q_median"]
        self.imputed_data['RECOLLECTION_SPEED'] = self.imputed_data["DEBTOR_DY_Q"] / self.imputed_data["DEBTOR_DY_Q_median"]

        # credit limit requests usage in the last 12 months
        self.imputed_data["clr_used_last_year"] = np.where(
            self.imputed_data["number_clr"] == 0,  # Condition (denominator = 0)
            0,  # Value if True (replace with 0 or np.nan)
            self.imputed_data["number_clr_last_12_months"] / self.imputed_data["number_clr"]  # Value if False
        )

        self.imputed_data.drop(self.imputed_data.filter(regex='_median$|_mean$').columns, axis=1, inplace=True)
        print(f"Feature Engineering completed: {self.imputed_data.shape[1]} features remaining.")


    def ScalingNormalization(self):
        """
        Function to perform scaling and normalization on the dataset.
        Args:
            imputed_data (pd.DataFrame): The dataset after performing feature engineering.
        """
        train_df = self.imputed_data[self.imputed_data['year'] < 2022].copy()
        test_df = self.imputed_data[self.imputed_data['year'] >= 2022].copy()

        cols = self.imputed_data.columns.to_list()
        categorical_features = ['Sector', 'REGION_GROUP', 'EX_POLICY_HOLDER', 
                                'BELONG_GROUP', 'is_company_italian']
        grade = ['VAL_GRD_C']
        counting_features = ['number_clr', 'number_clr_last_12_months']
        year = ['year', 'target']
        
        continous_features = [col for col in cols if col not in categorical_features + 
                              grade + counting_features + year]
        print(len(continous_features), "continous features")
        
        print("Starting Scaling and Normalization")
        # Normalize continuous features
        train_df_continous = train_df[continous_features]
        test_df_continous = test_df[continous_features]
        normalized_X_train, normalized_X_test = Data_Imputation.z_score_normalization(train_df_continous,
                                                                                      test_df_continous)
        normalized_X_train = pd.DataFrame(normalized_X_train, columns=continous_features).reset_index(drop=True)
        normalized_X_test = pd.DataFrame(normalized_X_test, columns=continous_features).reset_index(drop=True)
        normalized_data = pd.concat([normalized_X_train, normalized_X_test], axis=0).reset_index(drop=True)

        # Min-Max scaling for grade feature (range 0-10)
        X_train_grade = train_df[grade]
        X_test_grade = test_df[grade] 
        scaled_X_train_grade, scaled_X_test_grade = Data_Imputation.min_max_scaling(X_train_grade, X_test_grade)
        scaled_X_train_grade = pd.DataFrame(scaled_X_train_grade, columns=['scaled_grade']).reset_index(drop=True)
        scaled_X_test_grade = pd.DataFrame(scaled_X_test_grade, columns=['scaled_grade']).reset_index(drop=True)
        grade_data = pd.concat([scaled_X_train_grade, scaled_X_test_grade], axis=0).reset_index(drop=True)

        # Log transformation for counting features
        log_X_train_counting = np.log1p(train_df[counting_features])
        log_X_test_counting = np.log1p(test_df[counting_features])
        log_X_train_counting.reset_index(drop=True)
        log_X_test_counting.reset_index(drop=True)
        counting_data = pd.concat([log_X_train_counting, log_X_test_counting], axis=0).reset_index(drop=True)

        categorical_training_df = train_df[categorical_features].reset_index(drop=True)
        categorical_testing_df = test_df[categorical_features].reset_index(drop=True)
        categorical_data = pd.concat([categorical_training_df, categorical_testing_df], axis=0).reset_index(drop=True)

        target_train = train_df[year].reset_index(drop=True)
        target_test = test_df[year].reset_index(drop=True)
        target_data = pd.concat([target_train, target_test], axis=0).reset_index(drop=True)

        self.final_data = pd.concat([normalized_data, grade_data, 
                                     counting_data, categorical_data, 
                                     target_data], axis=1)
        
    def run_pipeline(self):
        """
        Run the entire data processing pipeline.
        """
        self.DataImputation()
        self.FeatureEngineering()
        self.ScalingNormalization()
        print("Data Processing Pipeline completed successfully.")

        # return self.final_data