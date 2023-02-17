import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve, f1_score, recall_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Define the LSTMClassifier class
class LSTMClassifier:
    def __init__(self, data_path):
        # Initialize the data path and variables to store the data, train and test sets, and model
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.model = None
        
    def load_data(self):
        # Load the data from the specified path
        self.data = pd.read_csv(self.data_path)
        
    def preprocess_data(self):
        # Split the data into features (X) and target (y)
        X = self.data.drop(["churn"], axis=1)
        y = self.data["churn"]
        
        # Split the data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize the features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Reshape the features to be 3D arrays suitable for input into an LSTM model
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1]))
        
    def build_model(self):
        # Create a sequential model with two LSTM layers, two dropout layers, and a dense output layer
        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=(1, self.X_train.shape[2]), return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(32))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation='sigmoid'))
        # Compile the model with the binary crossentropy loss function, the Adam optimizer, and accuracy metrics
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    def train_model(self, epochs, batch_size):
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_test, self.y_test))
        
    def evaluate_model(self):
        y_pred_prob = self.model.predict(self.X_test)
        y_pred_rounded = np.round(y_pred_prob)
        y_pred = y_pred_rounded.astype(int).ravel()
        
        # Evaluation metrics
        confusion_mat = confusion_matrix(self.y_test, y_pred)
        classification_re = classification_report(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)

        # Print the evaluation metrics
        print("Confusion Matrix:")
        print(confusion_mat)
        print("Classification Report:")
        print(classification_re)
        print("Accuracy:", accuracy)
        print("F1 Score:", f1)
        print("Recall:", recall)
        
lstm_classifier = LSTMClassifier("data/train.csv.gz")
lstm_classifier.load_data()
lstm_classifier.preprocess_data()
lstm_classifier.build_model()
lstm_classifier.train_model(epochs=50, batch_size=32)
lstm_classifier.evaluate_model()