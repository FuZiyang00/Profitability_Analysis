import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler


class Data_Imputation:

    def percentage_missing_df(data):
        missing_percentage = data.isnull().mean() * 100
        
        # Create a summary table
        nans_table = pd.DataFrame({
            'Column': missing_percentage.index,
            'Missing_Percentage': missing_percentage.values
        }).sort_values(by='Missing_Percentage', ascending=False).reset_index(drop=True)

        # Retain only columns with missing values (percentage > 0)
        nans_table = nans_table[nans_table['Missing_Percentage'] > 0].sort_values(by='Missing_Percentage', 
                                                                                ascending=False).reset_index(drop=True)
        
        return nans_table
    
    def turnover_proximity_imputer(row: pd.Series, X_train: pd.DataFrame, col_to_impute: str, tolerance=0.5) -> pd.Series:
        """
        Function to impute missing values based on THD_TRD_SEC_C and turnover amount.
        
        Parameters:
        - row: the row with the missing value to impute.
        - X_train: the training DataFrame used to find the closest match.
        - col_to_impute: the column with the missing value to impute.
        - tolerance: percentage (as a decimal) for the turnover proximity range (default 10%).
        
        Returns:
        - The closest non-NaN value from col_to_impute satisfying the criteria.
        """

        filtered_df = X_train[X_train['Sector'] == row['Sector']]

        if filtered_df.empty:
            # print(f"No data available for sector {row['Sector']}.")
            return np.nan
        
        # Ensure turnover proximity criteria
        turnover_lower = row['TURNOVER_M'] * (1 - tolerance)
        turnover_upper = row['TURNOVER_M'] * (1 + tolerance)
        
        turnover_filtered_df = filtered_df[
            (filtered_df['TURNOVER_M'] >= turnover_lower) & 
            (filtered_df['TURNOVER_M'] <= turnover_upper)
        ]
        
        # Select the closest non-NaN value in the col_to_impute
        valid_values = turnover_filtered_df[col_to_impute].dropna()
        # print(len(valid_values), "valid values found for imputation.")
        
        if not valid_values.empty:
            # Return the first value (or apply custom logic to select a value)
            return valid_values.iloc[0]

        # Fallback 2: Overall THD sector median (SAFELY)
        if not filtered_df.empty:
            non_nan_values = filtered_df[col_to_impute].dropna()
            if not non_nan_values.empty:
                return non_nan_values.median()  # Compute median only if non-NaN values exist
        else:
            return np.nan  # Return NaN if all sector values are NaN/empty
        

    def robust_scaling(X_train_numeric: pd.DataFrame, X_test_numeric: pd.DataFrame) -> tuple:
        """
        Robustly scale the training and test datasets using RobustScaler.
        """
        scaler = RobustScaler()
        scaled_X_train = scaler.fit_transform(X_train_numeric)
        scaled_X_test = scaler.transform(X_test_numeric)

        return scaled_X_train, scaled_X_test 
    
    
    def min_max_scaling(X_train_numeric: pd.DataFrame, X_test_numeric: pd.DataFrame) -> tuple:
        """
        Scale the training and test datasets using Min-Max scaling.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_X_train = scaler.fit_transform(X_train_numeric)
        scaled_X_test = scaler.transform(X_test_numeric)

        return scaled_X_train, scaled_X_test


    def z_score_normalization(X_train_numeric: pd.DataFrame, X_test_numeric: pd.DataFrame) -> tuple:
        """
        Normalize the training and test datasets using Z-score normalization.
        """
        scaler = StandardScaler()
        scaled_X_train = scaler.fit_transform(X_train_numeric)
        scaled_X_test = scaler.transform(X_test_numeric)

        return scaled_X_train, scaled_X_test
        
    