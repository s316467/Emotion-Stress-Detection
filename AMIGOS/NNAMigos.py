"""
Comprehensive Training Script:
- Deep Learning Models: CNN, RNN (with LSTM), and Transformer-based network.
- Classical Machine Learning Models: Logistic Regression, Random Forest, Decision Tree, 
  XGBoost, K-Nearest Neighbors, and a custom Gaussian Mixture Model (GMM) classifier.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Data Preprocessing
# -----------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

# Load the dataset
df = pd.read_csv('.\\Usuario5EEG.csv')
df.fillna(df.mean(), inplace=True)  # Replace NaNs with column means

# Separate features and labels
X = df.drop(columns=['Class'])
y = df['Class']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# For deep learning models (CNN, RNN, Transformer), we assume each sample is a sequence:
# Reshape the data to have one channel (samples, timesteps, channels)
X_train_seq = np.expand_dims(X_train, axis=-1)
X_test_seq = np.expand_dims(X_test, axis=-1)

# -----------------------
# Deep Learning Models
# -----------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Conv1D, MaxPooling1D, Flatten, LSTM,
                                     Input, LayerNormalization, MultiHeadAttention,
                                     Dropout, GlobalAveragePooling1D, Add)
from tensorflow.keras.optimizers import Nadam

# -----------------------
# 1. Convolutional Neural Network (CNN)
# -----------------------
cnn_model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_seq.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])

cnn_model.compile(optimizer=Nadam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\nTraining CNN Model...")
history_cnn = cnn_model.fit(X_train_seq, y_train, epochs=10, batch_size=30, 
                            validation_data=(X_test_seq, y_test), verbose=1)

# Evaluate CNN
y_pred_cnn = np.argmax(cnn_model.predict(X_test_seq), axis=-1)
accuracy_cnn = accuracy_score(y_test, y_pred_cnn)
recall_cnn = recall_score(y_test, y_pred_cnn, average='macro')
precision_cnn = precision_score(y_test, y_pred_cnn, average='macro')
f1_cnn = f1_score(y_test, y_pred_cnn, average='macro')
conf_matrix_cnn = confusion_matrix(y_test, y_pred_cnn)

print("\nCNN Performance:")
print(f"Accuracy: {accuracy_cnn:.4f}")
print(f"Recall: {recall_cnn:.4f}")
print(f"Precision: {precision_cnn:.4f}")
print(f"F1-Score: {f1_cnn:.4f}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_cnn, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('CNN Confusion Matrix')
plt.savefig('confusion_matrix_cnn.png')
plt.show()

# Plot CNN training curves
plt.figure(figsize=(8, 6))
plt.plot(history_cnn.history['loss'], label='Training Loss')
plt.plot(history_cnn.history['val_loss'], label='Validation Loss')
plt.title('CNN Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_cnn.png')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history_cnn.history['accuracy'], label='Training Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_cnn.png')
plt.show()

# -----------------------
# 2. Recurrent Neural Network (RNN) with LSTM
# -----------------------
rnn_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train_seq.shape[1], 1)),
    LSTM(32),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])

rnn_model.compile(optimizer=Nadam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\nTraining RNN Model...")
history_rnn = rnn_model.fit(X_train_seq, y_train, epochs=10, batch_size=30, 
                            validation_data=(X_test_seq, y_test), verbose=1)

# Evaluate RNN
y_pred_rnn = np.argmax(rnn_model.predict(X_test_seq), axis=-1)
accuracy_rnn = accuracy_score(y_test, y_pred_rnn)
recall_rnn = recall_score(y_test, y_pred_rnn, average='macro')
precision_rnn = precision_score(y_test, y_pred_rnn, average='macro')
f1_rnn = f1_score(y_test, y_pred_rnn, average='macro')
conf_matrix_rnn = confusion_matrix(y_test, y_pred_rnn)

print("\nRNN Performance:")
print(f"Accuracy: {accuracy_rnn:.4f}")
print(f"Recall: {recall_rnn:.4f}")
print(f"Precision: {precision_rnn:.4f}")
print(f"F1-Score: {f1_rnn:.4f}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rnn, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('RNN Confusion Matrix')
plt.savefig('confusion_matrix_rnn.png')
plt.show()

# Plot RNN training curves
plt.figure(figsize=(8, 6))
plt.plot(history_rnn.history['loss'], label='Training Loss')
plt.plot(history_rnn.history['val_loss'], label='Validation Loss')
plt.title('RNN Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_rnn.png')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history_rnn.history['accuracy'], label='Training Accuracy')
plt.plot(history_rnn.history['val_accuracy'], label='Validation Accuracy')
plt.title('RNN Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_rnn.png')
plt.show()

# -----------------------
# 3. Transformer-based Network
# -----------------------
def build_transformer_model(input_shape, head_size, num_heads, ff_dim, dropout=0.1):
    
    #Build a simple Transformer-based model.
    
    inputs = Input(shape=input_shape)
    
    # Transformer Encoder Block
    x = LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Add()([x, attention_output])
    
    x_norm = LayerNormalization(epsilon=1e-6)(x)
    ff_output = Dense(ff_dim, activation="relu")(x_norm)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = Dense(input_shape[-1])(ff_output)
    x = Add()([x, ff_output])
    
    # Pooling and final dense layers
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(4, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

transformer_model = build_transformer_model(input_shape=(X_train_seq.shape[1], 1),
                                            head_size=32, num_heads=2, ff_dim=32, dropout=0.1)

transformer_model.compile(optimizer=Nadam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\nTraining Transformer Model...")
history_transformer = transformer_model.fit(X_train_seq, y_train, epochs=10, batch_size=30,
                                              validation_data=(X_test_seq, y_test), verbose=1)

# Evaluate Transformer
y_pred_transformer = np.argmax(transformer_model.predict(X_test_seq), axis=-1)
accuracy_transformer = accuracy_score(y_test, y_pred_transformer)
recall_transformer = recall_score(y_test, y_pred_transformer, average='macro')
precision_transformer = precision_score(y_test, y_pred_transformer, average='macro')
f1_transformer = f1_score(y_test, y_pred_transformer, average='macro')
conf_matrix_transformer = confusion_matrix(y_test, y_pred_transformer)

print("\nTransformer Performance:")
print(f"Accuracy: {accuracy_transformer:.4f}")
print(f"Recall: {recall_transformer:.4f}")
print(f"Precision: {precision_transformer:.4f}")
print(f"F1-Score: {f1_transformer:.4f}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_transformer, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Transformer Confusion Matrix')
plt.savefig('confusion_matrix_transformer.png')
plt.show()

# Plot Transformer training curves
plt.figure(figsize=(8, 6))
plt.plot(history_transformer.history['loss'], label='Training Loss')
plt.plot(history_transformer.history['val_loss'], label='Validation Loss')
plt.title('Transformer Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_transformer.png')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history_transformer.history['accuracy'], label='Training Accuracy')
plt.plot(history_transformer.history['val_accuracy'], label='Validation Accuracy')
plt.title('Transformer Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_transformer.png')
plt.show()

# -----------------------
# Classical Machine Learning Models
# -----------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.mixture import GaussianMixture

# Custom GMM Classifier that fits one GMM per class.
class GMMClassifier:
    def __init__(self, n_components=1, covariance_type='full', random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.gmms_ = {}
        for cls in self.classes_:
            gmm = GaussianMixture(n_components=self.n_components, 
                                  covariance_type=self.covariance_type, 
                                  random_state=self.random_state)
            gmm.fit(X[y == cls])
            self.gmms_[cls] = gmm
        return self
        
    def predict(self, X):
        scores = np.array([self.gmms_[cls].score_samples(X) for cls in self.classes_]).T
        return self.classes_[np.argmax(scores, axis=1)]

# Define classifiers in a dictionary
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gaussian Mixture Model": GMMClassifier(n_components=4, covariance_type='full', random_state=42)
}

ml_stats_list = []

for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = clf.predict(X_test)
    
    # Compute performance metrics
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average='macro')
    prec = precision_score(y_test, y_pred, average='macro')
    f1_val = f1_score(y_test, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\n{name} Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1-Score: {f1_val:.4f}")
    
    ml_stats_list.append({
        "Model": name,
        "Accuracy": acc,
        "Recall": rec,
        "Precision": prec,
        "F1-Score": f1_val
    })
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'{name} Confusion Matrix')
    filename = f'confusion_matrix_{name.replace(" ", "_").lower()}.png'
    plt.savefig(filename)
    plt.show()

# Save classical ML models' statistics to CSV
ml_stats_df = pd.DataFrame(ml_stats_list)
ml_stats_df.to_csv('classical_ml_statistics.csv', index=False)
