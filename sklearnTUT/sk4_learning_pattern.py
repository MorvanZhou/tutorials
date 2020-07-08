# View more python learning tutorial on my Youtube and Youku channel!!!

# 网站：https://morvanzhou.github.io/tutorials/machine-learning/sklearn/
# 视频：https://www.bilibili.com/video/BV1xW411Y7Qd?p=5

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print(iris_X[:2, :])
print(iris_y)

X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=0.3)

print(y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(knn.predict(X_test))
print(y_test)
