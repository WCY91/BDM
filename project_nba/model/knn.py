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
import os
import pickle
from sklearn.metrics import accuracy_score
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
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
output_dir = "weights/"
os.makedirs(output_dir, exist_ok=True)

# 保存模型的函數
def save_model(model, filename):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filepath}")

# 記錄訓練過程的結果
def log_training_results(log_file, data):
    filepath = os.path.join(output_dir, log_file)
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Training log saved to {filepath}")

# KNN 訓練與最佳 K 的選擇
def decide_best_k(num_class, X_train, X_test, y_train, y_test):
    test_score = []
    for i in range(2, num_class + 1):
        neigh = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
        test_score.append(accuracy_score(y_test, neigh.predict(X_test)))
    return test_score.index(max(test_score)) + 2

num_class = len(set(money_y))
k = decide_best_k(num_class, X_train, X_test, y_train, y_test)
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

# 預測與記錄準確率
yhat = neigh.predict(X_test)
train_accuracy = accuracy_score(y_train, neigh.predict(X_train))
test_accuracy = accuracy_score(y_test, yhat)

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# 保存模型權重
save_model(neigh, f"knn_model_k{k}.pkl")

# 保存訓練記錄
training_results = {
    "k": [k],
    "train_accuracy": [train_accuracy],
    "test_accuracy": [test_accuracy]
}
log_training_results("knn_training_results.csv", training_results)
