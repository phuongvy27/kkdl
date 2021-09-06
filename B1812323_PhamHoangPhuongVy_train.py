import pandas as pd 
import numpy as np 
from sklearn import tree

iris = pd.read_csv('iris.csv')
# print(iris)

variety_mappings = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
iris = iris.replace(['Setosa', 'Versicolor' , 'Virginica'], [0, 1, 2])
#print(iris)

X = iris.iloc[:, 0:-1] 
y = iris.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = tree.DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# print(y_pred)
# # Tính độ chính xác
# print("Do chinh xac cua mo hinh voi nghi thuc kiem tra hold-out: %.3f" %
# accuracy_score(y_test, y_pred))
def classify(a, b, c, d):
    arr = np.array([a, b, c, d]) 
    arr = arr.astype(np.float64) 
    query = arr.reshape(1, -1) 
    prediction = variety_mappings[model.predict(query)[0]]
    return prediction
# Hiển thị cây
# tree.plot_tree(model.fit(X, y))
# plt.show()
