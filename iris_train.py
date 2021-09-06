import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

#Đọc dữ liệu từ file csv
dt = pd.read_csv("iris.csv")

#print(dt.head())

#Lấy dữ liệu
X = np.array(dt.iloc[:,0:4])
y = np.array(dt.iloc[:, 4:5])
#print(X)

#Xử lý dữ liệu -> chuyển cột nhãn sang dạng 0: Setosa, 1: Versicolor, 2: Virginica
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y = lb.fit_transform(y.ravel())

#print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=9)

clf = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
clf.fit(X_train, y_train)

#prediction = clf.predict(X_test)
#prediction = clf.predict([[6,3.4,4.5,1.6]])
#print(prediction)

#print("Accuracy: %.3f" %clf.score(X_test, y_test))

pickle.dump(clf, open('iris.pkl', 'wb'))
