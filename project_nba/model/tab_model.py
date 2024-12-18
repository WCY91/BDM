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
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, TFBertModel
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
import joblib


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 設定TensorFlow僅使用第一個GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)


salaries_data = pd.read_csv("cleaned_dataset/salaries_and_scores.csv")
salaries_data=salaries_data.drop('playerName',axis=1)
time_y  = salaries_data["MP"]
salaries_data.iloc[:, 0:17] = StandardScaler().fit_transform(salaries_data.iloc[:, 0:17])
print(salaries_data)

X = salaries_data[['PTS', 'PF', 'TOV', 'AST', 'STL','BLK','TRB','FG','FGA']].values
X = normalize(X, norm="l1")
y= time_y / 4 #make time label
X_train, X_test, y_train, y_test= train_test_split(
    X, y, test_size=0.2, random_state=3
)
