from __future__ import print_function
from sklearn import datasets
from sklearn.linear_model import LinearRegression

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)

print(model.predict(data_X[:4, :]))

# model.coef_ 和 model.intercept_ 属于 Model 的属性，
# 例如对于 LinearRegression 这个模型，这两个属性分别输出模型的斜率k和截距b（与y轴的交点）
print(model.coef_)
print(model.intercept_)

print(model.get_params())  # 取出之前定义的参数
print(model.score(data_X, data_y))  # 对 Model 用 R^2 的方式进行打分，输出精确度
# R^2: coefficient of determination, 决定系数
