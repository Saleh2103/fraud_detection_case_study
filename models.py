from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from colorama import init, Fore, Back, Style
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

# Initialize colorama
init()

class SMOTEHelper:
    @staticmethod
    def apply_smote(X, y):
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled

class LogisticRegressionModel:
    def __init__(self, dataset, smote_applied=False):
        self.dataset = dataset
        self.X = self.dataset.drop('fraud', axis=1)
        self.y = self.dataset['fraud']
        if smote_applied:
            self.X, self.y = SMOTEHelper.apply_smote(self.X, self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = LogisticRegression(max_iter=1000)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(Fore.BLUE + "Logistic Regression Model Accuracy:", accuracy)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        print(Fore.YELLOW + "Logistic Regression Model Confusion Matrix:")
        print(conf_matrix)
        class_report = classification_report(self.y_test, y_pred)
        print(Fore.GREEN + "Logistic Regression Model Classification Report:")
        print(class_report)
        # Reset the color
        print(Style.RESET_ALL + "")

class GradientBoostingModel:
    def __init__(self, dataset, smote_applied=False):
        self.dataset = dataset
        self.X = self.dataset.drop('fraud', axis=1)
        self.y = self.dataset['fraud']
        if smote_applied:
            self.X, self.y = SMOTEHelper.apply_smote(self.X, self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(Fore.BLUE + "Gradient Boosting Model Accuracy:", accuracy)
        # Reset the color
        print(Style.RESET_ALL + "")

class RandomForestModel:
    def __init__(self, dataset, smote_applied=False):
        self.dataset = dataset
        self.X = self.dataset.drop('fraud', axis=1)
        self.y = self.dataset['fraud']
        if smote_applied:
            self.X, self.y = SMOTEHelper.apply_smote(self.X, self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        print(Fore.BLUE + "Random Forest Model Accuracy:", accuracy)
        print(Fore.GREEN + "Random Forest Model Classification Report:")
        print(classification_report(self.y_test, predictions))
        print(Style.RESET_ALL + "")

class GatedRecurrentUnitModel:
    def __init__(self, dataset):
        self.dataset = dataset
        self.numerical_features = self.dataset.drop(columns=['fraud'])
        self.target_labels = self.dataset['fraud'].values
        self.X_numerical = self.numerical_features.values
        self.X_customer = self.dataset['customer_encoded'].values.reshape(-1, 1)
        self.X_merchant = self.dataset['merchant_encoded'].values.reshape(-1, 1)
        self.max_customer_length = len(self.dataset['customer_encoded'].unique())
        self.max_merchant_length = len(self.dataset['merchant_encoded'].unique())
        self.model = self.build_model()

    def build_model(self):
        numerical_input = tf.keras.Input(shape=(self.numerical_features.shape[1],), name='numerical_input')
        customer_input = tf.keras.Input(shape=(1,), name='customer_input')
        merchant_input = tf.keras.Input(shape=(1,), name='merchant_input')
        
        embedding_customer = tf.keras.layers.Embedding(input_dim=self.max_customer_length, output_dim=100)(customer_input)
        embedding_merchant = tf.keras.layers.Embedding(input_dim=self.max_merchant_length, output_dim=100)(merchant_input)
        
        gru_output_customer = tf.keras.layers.GRU(64)(embedding_customer)
        gru_output_merchant = tf.keras.layers.GRU(64)(embedding_merchant)
        
        concatenated = tf.keras.layers.concatenate([numerical_input, gru_output_customer, gru_output_merchant])
        
        dense1 = tf.keras.layers.Dense(128, activation='relu')(concatenated)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
        
        output = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)
        
        model = tf.keras.Model(inputs=[numerical_input, customer_input, merchant_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        X_numerical_train, self.X_numerical_test, \
        X_customer_train, self.X_customer_test, \
        X_merchant_train, self.X_merchant_test, \
        y_train, self.y_test = train_test_split(self.X_numerical, self.X_customer, self.X_merchant, self.target_labels, test_size=0.2, random_state=42)
        
        self.history = self.model.fit([X_numerical_train, X_customer_train, X_merchant_train], y_train,
                            epochs=10, batch_size=32, validation_split=0.2)
    
    def evaluate(self):
        loss, accuracy = self.model.evaluate([self.X_numerical_test, self.X_customer_test, self.X_merchant_test], self.y_test)
        print(Fore.RED + "Test Loss:", loss)
        print(Fore.BLUE + "Test Accuracy:", accuracy)
