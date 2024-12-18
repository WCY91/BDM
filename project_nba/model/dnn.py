from sklearn import metrics
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.metrics  import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout
from keras.models import Sequential,Model
from keras.layers import Dense,Input,Add,LeakyReLU,LayerNormalization,BatchNormalization
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import os

salaries_data = pd.read_csv("cleaned_dataset/salaries_and_scores.csv")
salaries_data=salaries_data.drop('playerName',axis=1)
time_y  = salaries_data["MP"]
salaries_data.iloc[:, 0:17] = StandardScaler().fit_transform(salaries_data.iloc[:, 0:17])
print(salaries_data)

X = salaries_data[['PTS', 'PF', 'TOV', 'AST', 'STL','BLK','TRB','FG','FGA']].values
X = normalize(X, norm="l1")
money_y= y  = salaries_data["target"]
time_y= time_y / 4 #make time label
X_train, X_test, y_train, y_test, time_y_train, time_y_test = train_test_split(
    X, y, time_y, test_size=0.2, random_state=3
)

#use regression model to classify the target label https://arxiv.org/pdf/2302.13386 nba2vec model 
def classification_model(num_classes):
  model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],)),
    LeakyReLU(alpha=0.1),
    Dropout(0.2),
    Dense(64),
    LeakyReLU(alpha=0.1),
    Dropout(0.2),
    Dense(32, name="bottleneck"),  
    LeakyReLU(alpha=0.1),
  ])
  model.add(Dense(32))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Dense(64))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Dense(128))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Dropout(0.2)),
  model.add(Dense(64))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Dense(32))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Dense(num_classes, activation="softmax"))

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def classification_model_with_residual(num_classes):
  inputs = Input(shape=(X_train.shape[1],))
  x = Dense(128, activation='relu')(inputs)
  x = Dropout(rate=0.2)(x)
  residual = Dense(128, activation='relu')(x) 
  residual = Dense(128, activation='relu')(residual)
  x = Add()([x, residual])  
  x = Dropout(rate=0.2)(x)
  residual = Dense(128, activation='relu')(x)  
  residual = Dense(128, activation='relu')(residual)
  x = Add()([x, residual])  
  x = Dropout(rate=0.2)(x)
  x = Dense(64, activation='relu')(x)
  x = Dense(32, activation='relu')(x)
  x = Dense(64,activation="relu")(x)
  x = Dense(128,activation="relu")(x)
  x = Dropout(rate=0.2)(x)
  x = Dense(64, activation="relu")(x)
  x = Dense(32, activation= "relu") (x)
  outputs = Dense(num_classes, activation='softmax')(x)
  model = Model(inputs, outputs)
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def train_model(model,x_train,x_test,y_train,y_test,target_name):
  model = model
  history = model.fit(x_train,y_train,epochs=100,validation_split=0.1,batch_size=32)
  loss,accuracy = model.evaluate(x_test,y_test)
  plt.figure(figsize=(10, 6))
  plt.plot(history.history['accuracy'], label='Baseline Training Accuracy', linestyle='--')
  plt.plot(history.history['val_accuracy'], label='Baseline Validation Accuracy', linestyle='--')
  
  plt.xlabel('Epochs')
  plt.ylabel(f'{target_name} Accuracy')
  plt.legend()
  plt.title(f'Model Comparison: {target_name} Accuracy')
  plt.grid(True)
  plt.savefig(f"{target_name}.jpg")
  model.save_weights(f"weights/{target_name}_baseline_weights.h5")
  print(loss)
  print(accuracy)



num_classes = 25
y_train = tf.keras.utils.to_categorical(y_train,num_classes=num_classes)
y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
model = classification_model(num_classes)
res_model = classification_model_with_residual(num_classes)
train_model(model,X_train,X_test,y_train,y_test_categorical,"money")
train_model(res_model,X_train,X_test,y_train,y_test_categorical,"res_money")

time_classes = len(set(time_y))
model = classification_model(time_classes)
res_model = classification_model_with_residual(time_classes)
time_y_train = tf.keras.utils.to_categorical(time_y_train,num_classes= time_classes)
time_y_test_categorical = tf.keras.utils.to_categorical(time_y_test, num_classes=time_classes)
train_model(model,X_train,X_test,time_y_train,time_y_test_categorical,"time")
train_model(res_model,X_train,X_test,time_y_train,time_y_test_categorical,"res_time")


# # all time predict
# all_time_salaries_scores_data = pd.read_csv("cleaned_dataset/all_time_salaries_and_scores.csv")
# all_time_salaries_scores_data=all_time_salaries_scores_data.drop(['playerName'],axis=1)
# time_y  = all_time_salaries_scores_data["MP"]
# salaries_data.iloc[:, 0:] = StandardScaler().fit_transform(salaries_data.iloc[:, 0:])

# X = all_time_salaries_scores_data[['seasonStartYear','PTS', 'PF', 'TOV', 'AST', 'STL','BLK','TRB','FG','FGA']].values
# X = normalize(X, norm="l1")
# time_y= time_y / 4 
# X_train, X_test, y_train, y_test= train_test_split(
#   X, time_y, test_size=0.2, random_state=3
# )
# time_classes = len(set(time_y))
# time_y_train = tf.keras.utils.to_categorical(y_train,num_classes= time_classes)
# time_y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=time_classes)
# predict_target(time_classes,X_train,X_test,time_y_train,time_y_test_categorical,"all__time")