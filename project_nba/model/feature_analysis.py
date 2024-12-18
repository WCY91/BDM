from sklearn import metrics
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.metrics  import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import sklearn.tree as tree
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics  import mean_squared_error

# Use the decision tree to classify and feature importance
salaries_data = pd.read_csv("C:\\Users\\0524e\\OneDrive\\文件\\GitHub\\BDM\\project_nba\\cleaned_dataset\\salaries_and_scores.csv")
X_linear = salaries_data["MP"].values
y_linear = salaries_data["inflationAdjSalary"].values
feature_names = ['PTS', 'PF', 'TOV', 'AST', 'STL', 'BLK', 'TRB', 'FG', 'FGA', 'MP']
X = salaries_data[feature_names].values
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
y = salaries_data["target"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=3)

nba_tree = DecisionTreeClassifier(criterion="entropy",max_depth=5)
nba_tree.fit(X_train,y_train)

pred = nba_tree.predict(X_test)
print (pred[0:5])
print (y_test[0:5])
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, pred))
tree.plot_tree(nba_tree)
plt.show()

importances = nba_tree.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(feature_names, importances)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('NBA Features ')
plt.show()
plt.scatter(X[:, 9], y, color='blue')  
plt.xlabel('MP')
plt.ylabel('Target')
plt.title('MP and Target relation')
plt.show()
# by use the decision tree most importance to draw the relation between the money and mp

model2 = LinearRegression()
X_linear = X_linear.reshape(-1, 1)
model2.fit(X_linear, y_linear)
plt.scatter(X_linear, y_linear,color='b')
plt.title("Use importance to fit the linear regression")
plt.plot(X_linear, model2.predict(X_linear),color='k')
plt.show()
print("y_linear min:", y_linear.min())
print("y_linear max:", y_linear.max())
print("y_linear std:", y_linear.std())

# Use the pca to decide feature importance
def get_feature_name_pca(component_weights,feature_names):
    feature_weights_mapping = {}
    for i, component in enumerate(component_weights):
        component_feature_weights = zip(feature_names, component)
        feature_weights_mapping[f"Component {i+1}"] = sorted(
        component_feature_weights, key=lambda x: abs(x[1]), reverse=True)
    return feature_weights_mapping

n = 3
all_feature_names = ['PTS', 'PF', 'TOV', 'AST', 'STL', 'BLK', 'TRB', 'FG', 'FGA', 'MP',"3P","3PA","FT","FTA","ORB","DRB","TRB"]
X = salaries_data[feature_names].values
pca = PCA(n_components=n)
principal_components = pca.fit_transform(X)
print("PCA scores", pca.explained_variance_ratio_)
component_weights = pca.components_
feature_mapping = get_feature_name_pca(component_weights,feature_names)
print("Feature names contributing to Principal Components")
for feature, weight in feature_mapping .items():
  print(f"{feature}: {weight}")

# Use the random forest to decide feature importance
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
importance = clf.feature_importances_
plt.bar(range(X.shape[1]), importance)
plt.xticks(range(X.shape[1]), feature_names, rotation=90)
plt.title("Feature Importance in Random Forest")
plt.show()



