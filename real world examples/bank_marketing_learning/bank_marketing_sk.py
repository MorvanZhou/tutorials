# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.

This data set is from: http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
Which is a real bank marketing data set. The required data are included in this example folder. You can download and
practice by yourself.

The 'bank-full.csv' data set has:
1) 17 inputs features (age, job, marital, education, default, balance, housing, loan,
   contact, day, month, duration, campaign, pdays, previous, poutcome);
2) 1 output (The answer yes or no to deposit to the bank); and
3) 45211 samples

We will use this data set for training and testing.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split, ShuffleSplit, cross_val_score
from sklearn import preprocessing
import matplotlib.pyplot as plt


def feature_utility(data, selected_feature_name, target_name):
    target_classes = data[target_name].unique()
    feature_classes = data[selected_feature_name].unique()
    indices = np.arange(len(feature_classes))
    percentages = np.zeros((len(target_classes), len(feature_classes)))
    for j, feature_class in enumerate(feature_classes):
        particular_feature = data[selected_feature_name][data[selected_feature_name] == feature_class]
        feature_total = len(particular_feature)
        for i, target_class in enumerate(target_classes):
            class_count = len(particular_feature[data[target_name] == target_class])
            percentage = class_count/feature_total
            percentages[i, j] = percentage

    colors = ['r', 'b', 'g']
    width = 1
    bars = []
    for i in range(len(target_classes)):
        c_number = int(i % len(colors))
        color = colors[c_number]
        if i == 0:
            bar = plt.bar(indices, percentages[i, :], width, color=color)
        else:
            bar = plt.bar(indices, percentages[i, :], width, color=color, bottom=percentages[:i, :].sum(axis=0))
        bars.append(bar)

    plt.xticks(indices + width/2, feature_classes)
    plt.ylabel('Percentage')
    plt.xlabel(selected_feature_name)
    plt.legend([bar[0] for bar in bars], target_classes, loc='best')
    plt.show()

def encode_label(data):
    la_en = preprocessing.LabelEncoder()
    for col in ['job', 'marital', 'education', 'default', 'housing', 'loan',
                'contact', 'month', 'poutcome', 'y']:
        data[col] = bank_data[col].astype('category')
        data[col] = la_en.fit_transform(bank_data[col])
    return data

dataset_path = ['bank.csv', 'bank-full.csv']
bank_data = pd.read_csv(dataset_path[1], sep=';')
print(bank_data.head())

# good categorical features: job, marital, education, housing, loan, contact, month, poutcome
# bad categorical features: default
# feature_utility(bank_data, 'housing', 'y')

bank_data = encode_label(bank_data)
# print(bank_data.dtypes)
# print(bank_data.head())

X_data, y_data = bank_data.iloc[:, :-1], bank_data.iloc[:, -1]
# show the percentage of answer yes and no.
answer_no, answer_yes = y_data.value_counts()
print('Percentage of answering no: ', answer_no/(answer_no+answer_yes))

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data,
    test_size=0.2)

dt_clf = DecisionTreeClassifier(class_weight='balanced',)
rf_clf = RandomForestClassifier(class_weight='balanced')
# randomize the data, and run the cross validation for 5 times
cv = ShuffleSplit(X_data.shape[0], n_iter=5,
        test_size=0.3, random_state=0)
print(cross_val_score(dt_clf, X_data, y_data, cv=cv, scoring='f1').mean())
print(cross_val_score(rf_clf, X_data, y_data, cv=cv, scoring='f1').mean())

# dt_clf.fit(X_train, y_train)
# print(dt_clf.score(X_test, y_test))
# rf_clf.fit(X_train, y_train)
# print(rf_clf.score(X_test, y_test))

# print(rf_clf.predict(X_test.iloc[10, :][np.newaxis, :]))
# print(y_test.iloc[10])
