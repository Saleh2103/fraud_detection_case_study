import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from colorama import init, Fore, Back, Style

# Initialize colorama
init()

class FeatureEngineering:
    
    def __init__(self):
        pass
    
    @staticmethod
    def one_hot_encoding(dataset, columns):
        """
        Perform one-hot encoding for specified columns in the dataset.
        
        Parameters:
            dataset (DataFrame): The DataFrame containing the data.
            columns (list): A list of column names for which one-hot encoding should be applied.
            
        Returns:
            DataFrame: The dataset with one-hot encoded columns.
        """
        encoded_dataset = dataset.copy()  # Create a copy of the dataset to avoid modifying the original
        
        # Perform one-hot encoding for each specified column
        for column in columns:
            # Get one-hot encoding for the column
            one_hot_encoded = pd.get_dummies(encoded_dataset[column], prefix=column)
            
            # Drop the original column from the dataset
            encoded_dataset = encoded_dataset.drop(column, axis=1)
            
            # Concatenate the one-hot encoded column with the dataset
            encoded_dataset = pd.concat([encoded_dataset, one_hot_encoded], axis=1)
            
            # Convert boolean values in the one-hot encoded columns to integers (0 or 1)
            encoded_dataset[one_hot_encoded.columns] = encoded_dataset[one_hot_encoded.columns].astype(int)
            print(Fore.CYAN + 'One-Hot encoding for age, gender, and category columns')
            print(Style.RESET_ALL + "")
        
        return encoded_dataset
    
    @staticmethod
    def frequency_encoding(dataset, columns):
        """
        Perform frequency encoding for specified columns in the dataset.
        
        Parameters:
            dataset (DataFrame): The DataFrame containing the data.
            columns (list): A list of column names for which frequency encoding should be applied.
            
        Returns:
            DataFrame: The dataset with frequency encoded columns.
        """
        encoded_dataset = dataset.copy()  # Create a copy of the dataset to avoid modifying the original
        
        for column in columns:
            # Calculate the frequency of each category in the column
            frequency = encoded_dataset[column].value_counts(normalize=True)
            
            # Replace each category with its frequency
            encoded_dataset[column] = encoded_dataset[column].map(frequency)
        print(Fore.CYAN + 'Frequency encoding for customer and merchant columns')
        print(Style.RESET_ALL + "")
            
        return encoded_dataset
    
    @staticmethod
    def target_encoding(dataset, target_column, categorical_column):
        """
        Perform target encoding for a categorical column in the dataset.
        
        Parameters:
            dataset (DataFrame): The DataFrame containing the data.
            target_column (str): The name of the target variable column.
            categorical_column (str): The name of the categorical column to encode.
            
        Returns:
            DataFrame: The dataset with the categorical column target encoded.
        """
        encoded_dataset = dataset.copy()  # Create a copy of the dataset to avoid modifying the original
        
        # Calculate the mean of the target variable for each category in the categorical column
        target_means = dataset.groupby(categorical_column)[target_column].mean()
        
        # Map the means to the categorical column
        encoded_dataset[categorical_column] = encoded_dataset[categorical_column].map(target_means)
        print(Fore.CYAN + 'Target encoding for customer and merchant columns')
        print(Style.RESET_ALL + "")
        
        return encoded_dataset
    
    @staticmethod
    def tokenize_and_standardize(dataset):
        """
        Tokenizes the 'customer' and 'merchant' columns using LabelEncoder and standardizes the 'step' and 'amount' columns.

        Parameters:
        - dataset (DataFrame): The DataFrame containing the 'customer', 'merchant', 'step', and 'amount' columns.

        Returns:
        - Processed dataset.
        """
        # Tokenization using LabelEncoder
        label_encoder = LabelEncoder()
        dataset['customer_encoded'] = label_encoder.fit_transform(dataset['customer'])
        dataset['merchant_encoded'] = label_encoder.fit_transform(dataset['merchant'])
        print(Fore.CYAN + 'customer and merchant columns are labeled')
        print(Style.RESET_ALL + "")
        dataset.drop(columns=['customer','merchant'], inplace=True)
        
        # Applying standardization for step and amount columns
        scaler = StandardScaler()
        
        # Convert the 'step' and 'amount' columns to float64
        dataset[['step', 'amount']] = dataset[['step', 'amount']].astype('float64')
        dataset[['step', 'amount']] = scaler.fit_transform(dataset[['step', 'amount']])
        print(Fore.CYAN + 'step and amount are standardized')
        print(Style.RESET_ALL + "")
        return dataset

class DatasetCleaning:
    def __init__(self):
        pass
    
    @staticmethod
    def clean(dataset):
        # Clean columns from the quotation
        dataset[['customer','age','gender','merchant','category']] = dataset[['customer','age','gender','merchant','category']].map(lambda x: x.replace("'", ""))
        print(Fore.CYAN + 'single quotations in columns have been cleaned')
        print(Style.RESET_ALL + "")

        # Drop zipcodeOri and zipMerchant columns as they have one unique value
        dataset = dataset.drop(columns=['zipcodeOri', 'zipMerchant'])
        print(Fore.CYAN + 'zipcodeOri and zipMerchant have been dropped from the dataset')
        print(Style.RESET_ALL + "")

        # Drop any row with missing value in fraud column because it's useless
        dataset.dropna(subset=['fraud'], inplace=True)
        print(Fore.CYAN + 'Dropped any missing value in the target column')
        print(Style.RESET_ALL + "")

        # Fill missing values in 'amount' column with mean
        grouped_mean = dataset.groupby(['customer', 'category'])['amount'].transform('mean')
        dataset['amount'] = dataset['amount'].fillna(grouped_mean)
        print(Fore.CYAN + 'Filled any missing value in amount column with MEAN')
        print(Style.RESET_ALL + "")

        # Define categorical columns
        categorical_columns = ['customer', 'age', 'gender', 'merchant', 'category']

        # Mode for categorical columns with dependencies to reduce cardinality
        for col in categorical_columns:
            if col != 'customer':
                # Make a copy of the categorical columns list without the current column
                group_cols = categorical_columns.copy()
                group_cols.remove(col)
                
                # Calculate mode based on remaining categorical columns
                mode = dataset.groupby(group_cols)[col].transform(lambda x: x.mode().iloc[0])
                
                # Fill missing values with mode
                dataset[col] = dataset[col].fillna(mode)
        print(Fore.CYAN + 'Filled any missing value in categorical columns with MODE')
        print(Style.RESET_ALL + "")
        return dataset
