Summary:
Fraud detection is a critical aspect of ensuring the integrity and security of financial systems. In this analysis, various machine learning (ML) models were evaluated for their effectiveness in detecting fraudulent activities. 
The models were trained and tested using datasets that underwent both One-Hot encoding and Target encoding, with and without Synthetic Minority Over-sampling Technique (SMOTE) applied to address class imbalance.
Logistic Regression:
•	With SMOTE: Achieved an accuracy of 99.62%, exhibiting robust performance with precision, recall, and F1-score all at 1.00, indicating excellent detection capabilities for both fraudulent and non-fraudulent transactions.
•	Without SMOTE: Despite a slightly higher accuracy of 99.65%, the model's performance on detecting fraudulent transactions decreased, as evidenced by a lower recall and F1-score for the minority class.
Gradient Boosting Machine (GBM):
•	With SMOTE: Showcased a strong accuracy of 99.60%, indicating reliable performance in fraud detection.
•	Without SMOTE: Demonstrated a slightly higher accuracy of 99.69%, suggesting effective fraud detection capabilities even without oversampling.
Random Forest:
•	With SMOTE: Delivered an impressive accuracy of 99.82%, maintaining high precision, recall, and F1-score for both classes, reflecting robust fraud detection capabilities.
•	Without SMOTE: Despite a slightly lower accuracy of 99.71%, the model maintained solid performance in identifying fraudulent transactions, albeit with a slight decrease in recall compared to the SMOTE-enhanced version.
GRU Model:
•	Utilizing advanced techniques such as tokenization, standardization, and embedding, the GRU model achieved a commendable accuracy of 99.58%, underscoring its potential in fraud detection tasks.
Conclusion:
•	Among the evaluated models, Random Forest with SMOTE stands out as the most effective option, offering the highest accuracy and consistently robust performance in detecting fraudulent transactions. However, further fine-tuning and evaluation may be warranted to ensure optimal model performance and adaptability to evolving fraud patterns.
