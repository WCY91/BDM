from sklearn import metrics
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input, Dropout
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

salaries_data = pd.read_csv("cleaned_dataset/salaries_and_scores.csv")
salaries_data = salaries_data.drop('playerName', axis=1)

time_y = salaries_data["MP"]           
salaries_data.iloc[:, 0:17] = StandardScaler().fit_transform(salaries_data.iloc[:, 0:17])

X = salaries_data[['PTS', 'PF', 'TOV', 'AST', 'STL','BLK','TRB','FG','FGA']].values
X = normalize(X, norm="l1")

money_y = salaries_data["target"]

time_y = (time_y / 4).astype(int)

X_train, X_test, y_train, y_test, time_y_train, time_y_test, _, _ = train_test_split(
    X, money_y, time_y, salaries_data['MP'].values,
    test_size=0.2, random_state=3
)

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

def build_tabtransformer_classification(
    num_features,
    num_classes,             
    embed_dim=8,
    num_heads=2,
    ff_dim=8,
    num_transformer_blocks=2,
    mlp_units=[16, 8],
    dropout_rate=0.1,
    mlp_dropout=0.2
):
    inputs = layers.Input(shape=(num_features,))  

    x = layers.Dense(num_features * embed_dim)(inputs)
    x = layers.Reshape((num_features, embed_dim))(x)

    for _ in range(num_transformer_blocks):
        x = TransformerEncoder(embed_dim, num_heads, ff_dim, rate=dropout_rate)(x)

    x = layers.GlobalAveragePooling1D()(x)

    for units in mlp_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

num_features = X_train.shape[1]               
num_classes = len(np.unique(time_y_train))    
print("num_classes =", num_classes)

model = build_tabtransformer_classification(
    num_features=num_features,
    num_classes=num_classes,
    embed_dim=8,            
    num_heads=2,
    ff_dim=16,
    num_transformer_blocks=2,
    mlp_units=[32, 16],
    dropout_rate=0.1,
    mlp_dropout=0.1
)


model.compile(
    loss="sparse_categorical_crossentropy",  # time_y 是 integer label
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"]
)

history = model.fit(
    X_train, 
    time_y_train,               
    validation_split=0.2,
    epochs=50,
    batch_size=32
)


y_pred_proba = model.predict(X_test)               
y_pred_label = np.argmax(y_pred_proba, axis=1)     # 取最大機率類別

acc = accuracy_score(time_y_test, y_pred_label)
print("Accuracy on test data:", acc)
print("Classification Report:")
print(classification_report(time_y_test, y_pred_label))

conf_mat = confusion_matrix(time_y_test, y_pred_label)
print("Confusion Matrix:\n", conf_mat)


plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
