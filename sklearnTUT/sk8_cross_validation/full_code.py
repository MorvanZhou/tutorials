# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split  # 新版中由cross_validation改成了model_selection
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target

# test train split #
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(knn.score(X_test, y_test))

# this is cross_val_score #
from sklearn.model_selection import cross_val_score

knn = KNeighborsClassifier(n_neighbors=5)
# 使用K(5)折交叉验证模块
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
print(scores)
print(scores.mean())

# this is how to use cross_val_score to choose model and configs #
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # 一般来说平均方差(Mean squared error)会用于判断回归(Regression)模型的好坏，平均方差越低越好
    # scoring='mean_squared_error'将在0.20以后弃用，现在改成了scoring='neg_mean_squared_error'
    # loss = -cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error') # for regression
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')  # for classification
    k_scores.append(scores.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
