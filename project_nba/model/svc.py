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
import os
import pickle
from sklearn import metrics
import pandas as pd
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

# 訓練記錄保存文件夾
output_dir = "weights/"
os.makedirs(output_dir, exist_ok=True)

# 保存模型的函數
def save_model(model, filename):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filepath}")

# 保存訓練記錄
def log_training_results(log_file, data):
    filepath = os.path.join(output_dir, log_file)
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Training log saved to {filepath}")

# 使用 SVM 進行多分類
base_svm = SVC(probability=True, random_state=42)  # 加入 `random_state` 確保結果可重現
base_svm.fit(X_train, y_train)
test_pred = base_svm.predict(X_test)

# 計算準確率
train_accuracy = accuracy_score(y_train, base_svm.predict(X_train))
test_accuracy = accuracy_score(y_test, test_pred)
print(f"SVM Train Accuracy: {train_accuracy}")
print(f"SVM Test Accuracy: {test_accuracy}")

# 保存模型
save_model(base_svm, "svm_model.pkl")

# 保存訓練記錄
training_results = {
    "kernel": [base_svm.kernel],
    "C": [base_svm.C],
    "probability": [base_svm.probability],
    "random_state": [42],
    "train_accuracy": [train_accuracy],
    "test_accuracy": [test_accuracy]
}
log_training_results("svm_training_results.csv", training_results)
