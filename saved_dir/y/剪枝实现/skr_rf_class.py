import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle  # 可以输入两部分进行打乱   # ****


rf = RandomForestClassifier(oob_score=True)   # 表示所有参数都用默认的
train_df = pd.read_csv('sonar.csv', header=None)
# print(train_df.head())
label_df = train_df.pop(60).map({'M': 0, 'R': 1})  # 将标签取出，tain——df只剩下特征的  标签替换得放到这里 # *****
# train_df.drop(inndex, inplace=True)
# train = train_df.drop(inndex)   # 都表示删掉某个数据
train_df.drop(train_df[train_df[3]>=3].index, inplace=True)
train = train_df.values   # 将值取出  特征
label = label_df.values   # 标签

# print(label)

# rf.fit(train, label)
# # print(rf.oob_score_)

# train, label = shuffle(train, label, random_state=5)  # random_state 表示打乱程度，越大，打乱越大

# train_fea = train[:160]
# train_label = label[:160]
# test_fea = train[160:]
# test_label = label[160:]

# rf = RandomForestClassifier(oob_score=True)
# # 训练
# rf.fit(train_fea, train_label)
# # print(rf.oob_score_)
# pred_label = rf.predict(test_fea)
# # print(rf.predict(test_fea))
# # print('real_label:', test_label)
# # print[pred_label == test_label].count(True)/len(test_label))
# from sklearn.metrics import accuracy_score   # 计算正确率   # ******
# # print(accuracy_score(test_label, pred_label))

# pred_prob_label = rf.predict_proba(test_fea)       # 前面的是属于0的概率，可用于计算AUC值
# # print(pred_prob_label)

# # 求AUC 
# from sklearn.metrics import roc_auc_score
# print(roc_auc_score(test_label, pred_prob_label[:, -1]))


# # 利用交叉验证的方法
# from sklearn.model_selection import cross_val_score

# def rmse_cv(model, X_train, y):
#     rmse = cross_val_score(model, X_train, y, 
#                                 scoring='accuracy',    # 'roc_auc'
#                                 cv=6)
#     return rmse

# NE = range(1, 61, 5)
# MSE = range(2, 10, 2)

# # score = rmse_cv(RandomForestClassifier(n_estimators=10), train, label)
# # print(score)
# score = [rmse_cv(RandomForestClassifier(n_estimators=20, min_samples_split=mse), train, label).mean() for mse in MSE]
# print(score)
# import matplotlib.pyplot as plt
# plt.plot(MSE, score)
# plt.show()

rf = RandomForestClassifier(n_estimators=30, oob_score=True)
train_df = pd.read_csv('sonar.csv', header=None)
summary = train_df.describe()
print(summary)

# 对前十个进行标准化
for i in range(10):
    train_df.iloc[:,i] = (train_df.iloc[:,i] - summary.iloc[1, i])/summary.iloc[2, i]

import matplotlib.pyplot as plt

plt.boxplot(train_df.iloc[:,0:10].values)
plt.show()