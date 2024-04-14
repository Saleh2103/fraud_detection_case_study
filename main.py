import pandas as pd
import preprocessing
import models

# Reading the dataset
dataset = pd.read_csv('fraud.csv', quotechar="'")

cleaning = preprocessing.DatasetCleaning()
# Cleaning the dataset
dataset = cleaning.clean(dataset)

# Creating feature instance
featuring = preprocessing.FeatureEngineering()

# One-Hot encoding for age, gender, and category columns
dataset_one_hot = featuring.one_hot_encoding(dataset,['age', 'gender', 'category'])

# Target encoding for customer and merchant columns
dataset_hot_target = featuring.target_encoding(dataset_one_hot,'fraud','customer')
dataset_hot_target = featuring.target_encoding(dataset_hot_target,'fraud','merchant')

# Frequency encoding for customer and merchant columns
dataset_hot_frequency = featuring.frequency_encoding(dataset_one_hot,['customer','merchant'])

# Preparing GRU dataset
dataset_gru = dataset_one_hot.copy()
# Tokenization and standardization
dataset_gru = featuring.tokenize_and_standardize(dataset_gru)

print()  # Print empty line before
print("####" * 20)
print()  # Print empty line after

# Logistics Regression model with One-Hot encoding and Target encoding with SMOTE
logistic_regression_model = models.LogisticRegressionModel(dataset_hot_target,smote_applied = True)
print('The below Logistics Regression result is based on One-Hot encoding and Target encoding dataset with SMOTE:')
logistic_regression_model.train()
logistic_regression_model.evaluate()

print()  # Print empty line before
print("####" * 20)
print()  # Print empty line after

# Logistics Regression model with One-Hot encoding and Target encoding without SMOTE
logistic_regression_model = models.LogisticRegressionModel(dataset_hot_target,smote_applied = False)
print('The below Logistics Regression result is based on One-Hot encoding and Target encoding dataset without SMOTE:')
logistic_regression_model.train()
logistic_regression_model.evaluate()

print()  # Print empty line before
print("####" * 20)
print()  # Print empty line after

# Gradient Boosting model with One-Hot encoding and Target encoding with SMOTE
gbm_model = models.GradientBoostingModel(dataset_hot_target,smote_applied = True)
print('The below GBM result is based on One-Hot encoding and Target encoding dataset with SMOTE:')
gbm_model.train()
gbm_model.evaluate()

print()  # Print empty line before
print("####" * 20)
print()  # Print empty line after

# Gradient Boosting model with One-Hot encoding and Target encoding without SMOTE
gbm_model = models.GradientBoostingModel(dataset_hot_target,smote_applied = False)
print('The below GBM result is based on One-Hot encoding and Target encoding dataset without SMOTE:')
gbm_model.train()
gbm_model.evaluate()

print()  # Print empty line before
print("####" * 20)
print()  # Print empty line after

# Gradient Boosting model with One-Hot encoding and Frequency encoding with SMOTE
gbm_model = models.GradientBoostingModel(dataset_hot_frequency,smote_applied = True)
print('The below GBM result is based on One-Hot encoding and Frequency encoding dataset with SMOTE:')
gbm_model.train()
gbm_model.evaluate()

print()  # Print empty line before
print("####" * 20)
print()  # Print empty line after

# Gradient Boosting model with One-Hot encoding and Frequency encoding without SMOTE
gbm_model = models.GradientBoostingModel(dataset_hot_frequency,smote_applied = False)
print('The below GBM result is based on One-Hot encoding and Frequency encoding dataset without SMOTE:')
gbm_model.train()
gbm_model.evaluate()

print()  # Print empty line before
print("####" * 20)
print()  # Print empty line after

# Random Forest model with One-Hot encoding and Target encoding with SMOTE
random_forest_model = models.RandomForestModel(dataset_hot_target,smote_applied = True)
print('The below Random Forest result is based on One-Hot encoding and Target encoding dataset with SMOTE:')
random_forest_model.train()
random_forest_model.evaluate()

print()  # Print empty line before
print("####" * 20)
print()  # Print empty line after

# Random Forest model with One-Hot encoding and Target encoding without SMOTE
random_forest_model = models.RandomForestModel(dataset_hot_target,smote_applied = False)
print('The below Random Forest result is based on One-Hot encoding and Target encoding dataset without SMOTE:')
random_forest_model.train()
random_forest_model.evaluate()

print()  # Print empty line before
print("####" * 20)
print()  # Print empty line after

# Random Forest model with One-Hot encoding and Frequency encoding with SMOTE
random_forest_model = models.RandomForestModel(dataset_hot_frequency,smote_applied = True)
print('The below Random Forest result is based on One-Hot encoding and Frequency encoding dataset with SMOTE:')
random_forest_model.train()
random_forest_model.evaluate()

print()  # Print empty line before
print("####" * 20)
print()  # Print empty line after

# Random Forest model with One-Hot encoding and Frequency encoding without SMOTE
random_forest_model = models.RandomForestModel(dataset_hot_frequency,smote_applied = False)
print('The below Random Forest result is based on One-Hot encoding and Frequency encoding dataset without SMOTE:')
random_forest_model.train()
random_forest_model.evaluate()

print()  # Print empty line before
print("####" * 20)
print()  # Print empty line after

# Gated Recurrent Unit (GRU) model with One-Hot encoding and Frequency encoding
gru_model = models.GatedRecurrentUnitModel(dataset_gru)
print('The below GRU result is based on One-Hot encoding, Tokenization, standardization, and Embedding dataset:')
gru_model.train()
gru_model.evaluate()
