import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve, f1_score, recall_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Define the function to create the LSTM model outside the class
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Define the LSTMClassifier class
class LSTMClassifier:
    
    """
    
    Class Created by: Diego Gustavo Hurtado Olivares
    
    LSTMClassifier is a class that provides an easy-to-use interface for training, evaluating, 
    and tuning Long Short-Term Memory (LSTM) models for binary classification tasks.
    
    This class includes methods for preprocessing data, building and training the LSTM model,
    evaluating the model using various metrics, plotting the training history and ROC curve,
    predicting new instances, saving and loading the model, and obtaining feature importances 
    using permutation importance. It also provides support for hyperparameter tuning, K-fold
    cross-validation, and early stopping.
    
    Example usage:
    --------------
    
    # Initialize the LSTMClassifier with your data
    lstm_classifier = LSTMClassifier(data)
    
    # Preprocess the data
    lstm_classifier.preprocess_data()
    
    # Build the LSTM model
    lstm_classifier.build_model()
    
    # Train the model
    history = lstm_classifier.train_model(epochs=50, batch_size=32)
    
    # Evaluate the model
    lstm_classifier.evaluate_model()
    
    # Plot the training history and ROC curve
    lstm_classifier.plot_training_history(history)
    lstm_classifier.plot_roc_curve()
    
    # Predict new instances
    y_pred = lstm_classifier.predict(X_new)
    
    """
    
    def __init__(self, data):
        # Initialize the data path and variables to store the data, train and test sets, and model
        # self.data_path = data_path
        self.data = data
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
        
    def early_stopping(self, patience=10, restore_best_weights=True):
        return EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=restore_best_weights)
    
    def custom_metric(self, y_true, y_pred):
        # Example: Implement the balanced accuracy metric
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        return balanced_accuracy
    
        
    def evaluate_model(self, use_custom_metric=True):
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

        if use_custom_metric:
            custom_metric_value = self.custom_metric(self.y_test, y_pred)
            print("Custom Metric Value (Balanced Accuracy):", custom_metric_value)
        
    def plot_training_history(self, history):
        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 5))
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.figure(figsize=(12, 5))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def plot_roc_curve(self):
        y_pred_prob = self.model.predict(self.X_test).ravel()
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_prob)
        auc_score = auc(fpr, tpr)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()
        
    def predict(self, X):
        # Preprocess input
        X_scaled = self.scaler.transform(X)
        X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        # Make predictions
        y_pred_prob = self.model.predict(X_reshaped)
        y_pred_rounded = np.round(y_pred_prob)
        y_pred = y_pred_rounded.astype(int).ravel()
        return y_pred

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = load_model(model_path)

    def get_model_summary(self):
        self.model.summary()
        
    def tune_hyperparameters(self, param_grid, cv=5, search_type='grid', n_iter=None, random_state=42):
        input_shape = (1, self.X_train.shape[2])

        # Wrap the model for use with scikit-learn
        model = KerasClassifier(build_fn=lambda: create_lstm_model(input_shape), verbose=0)

        if search_type == 'grid':
            search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv)
        elif search_type == 'random':
            if n_iter is None:
                raise ValueError("n_iter must be specified for random search.")
            search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=cv,
                                        n_iter=n_iter, random_state=random_state)

        search_result = search.fit(self.X_train, self.y_train)

        # Print the best score and best parameters
        print("Best score: %f using %s" % (search_result.best_score_, search_result.best_params_))
        return search_result

    def k_fold_cross_validation(self, n_splits=5, epochs=50, batch_size=32):
        input_shape = (1, self.X_train.shape[2])

        # Define a function to create the model with the proper input shape
        def create_model():
            return create_lstm_model(input_shape)

        # Wrap the model for use with scikit-learn
        model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)

        # Perform k-fold cross-validation
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        results = cross_val_score(model, self.X_train, self.y_train, cv=kfold)

        # Print the mean and standard deviation of the cross-validation scores
        print("Cross-Validation Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    def model_to_json(self):
        model_json = self.model.to_json()
        return model_json

    def json_to_model(self, model_json):
        self.model = model_from_json(model_json)
        
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
        y_pred_prob = self.model.predict(X_reshaped)
        return y_pred_prob

    def train_val_split(self, val_size=0.1, random_state=42):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=val_size, random_state=random_state)

        # Reshape X_val using the same number of features as in X_train and X_test
        self.X_val = np.reshape(self.X_val, (self.X_val.shape[0], 1, self.X_train.shape[2]))


    def learning_rate_reduction(self, factor=0.1, patience=10, min_lr=1e-5):
        return ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience, min_lr=min_lr)

    def train_model(self, epochs, batch_size, use_early_stopping=True, use_lr_reduction=True):
        callbacks = []
        if use_early_stopping:
            callbacks.append(self.early_stopping())
        if use_lr_reduction:
            callbacks.append(self.learning_rate_reduction())

        history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(self.X_test, self.y_test), callbacks=callbacks)
        return history
    
    def get_feature_importance(self, X, y, n_repeats=10, random_state=42):
        # Wrap the predict_proba method for use with sklearn's permutation_importance function
        def predict_proba_wrapped(X):
            return self.predict_proba(X)

        # Compute the permutation importance
        result = permutation_importance(predict_proba_wrapped, X, y, n_repeats=n_repeats,
                                        random_state=random_state, n_jobs=-1)

        # Combine the feature importances and their names into a dataframe and sort by importance
        feature_importance = pd.DataFrame({'feature': self.data.drop(["churn"], axis=1).columns,
                                           'importance': result.importances_mean,
                                           'std': result.importances_std})

        feature_importance = feature_importance.sort_values(by='importance', ascending=False)

        return feature_importance
