import pandas as pd
# pd的横纵坐标索引反过来了 *********iloc与平时一样
import numpy as np
import matplotlib.pylab as plt

from sklearn.ensemble import RandomForestClassifier  
from sklearn import metrics  # 用于计算模型的正确率
from sklearn.model_selection import cross_val_score  # 用于后面的交叉验证
from sklearn.utils import shuffle  # 可以将输入的两部分同时进行打乱   # ****

# # python开发中经常遇到报错的情况，但是warning通常并不影响程序的运行，而且有时特别讨厌，下面方法可以忽略warning错误  # *****
# import warnings
# warnings.filterwarnings('ignore')

# # plt参数设置  # ***
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示符号

# # 读取数据
# data_df = pd.read_csv('sonar.csv', header=None, index_col=None)
# # print(data_df.head())

# # 调参步骤
# # 首先查看分类的分布情况
# # 查看特征属性
# # 查看连续特征的分布情况：箱线图  #******（用于删除异常点）
# # 查看特征之间的相关性-包括各个特征与标签的相关性  （特征选择：特征删除，特征组合）
# # 缺失值的填充
# # 模型调参



# # 查看分布情况
# grouped = data_df[60]   # 表示数据的第60列
# # print(grouped)
# grouped = data_df[60].groupby(data_df[60])
# # print(grouped)  # <pandas.core.groupby.SeriesGroupBy object at 0x7fe12daffd68>
# # print(grouped.count())   60
# #                         M    111
# #                         R     97
# #                         dtype: int64
# # print(grouped.count() / data_df[60].count())    # 60
#                                                 # M    0.533654
#                                                 # R    0.466346
#                                                 # dtype: float64



# # 找到字符型特征和数字特征
# # 数字特征
# numeric_feats = data_df.dtypes[data_df.dtypes != 'object'].index
# # print(numeric_feats)
# # print(len(numeric_feats))
# # 字符特征
# categorical_feats = data_df.dtypes[data_df.dtypes == 'object'].index
# # print(categorical_feats)
# # print(len(categorical_feats))

# # 对前十个特征进行标准化并查看箱线图
# summary = data_df.describe()
# # print(summary[0])   count    208.000000
# #                     mean       0.029164
# #                     std        0.022991
# #                     min        0.001500
# #                     25%        0.013350
# #                     50%        0.022800
# #                     75%        0.035550
# #                     max        0.137100
# #                     Name: 0, dtype: float64
# # print(summary[0][1])   #0.0291639423077
# col_num = len(data_df.columns)
# # print(col_num)   # 61
# for i in range(10):
#     data_df.iloc[:, i] = (data_df.iloc[:, i] - summary.iloc[1, i]) / summary.iloc[2, i]

# # plt.boxplot(data_df.iloc[:, :10].values)
# # plt.show()


# # 删除几个带极端值的样本    ******
# data_df.drop(data_df[data_df[0]>4].index, inplace=True)
# data_df.drop(data_df[data_df[1]>3].index, inplace=True)
# data_df.drop(data_df[data_df[2]>3.5].index, inplace=True)
# data_df.drop(data_df[data_df[3]>4].index, inplace=True)
# data_df.drop(data_df[data_df[4]>4].index, inplace=True)
# data_df.drop(data_df[data_df[5]>5].index, inplace=True)
# # plt.boxplot(data_df.iloc[:, :10].values)
# # plt.show()


# # 将字符改成类别标签
# data_df[60] = data_df[60].map({'M': 0, 'R': 1})
# # print(data_df)
# labels = data_df.pop(60)

# labels = labels.values
# fea = data_df.values

# # print(labels)
# # print(fea)

# X, y = shuffle(fea, labels, random_state=5)  # 同时打乱两个， random_state相当于seed *****
# # print(X, y)


# # 先观察模型性对训练集进行训练后的效果；不区分训练验证测试
# rf0 = RandomForestClassifier(oob_score=True, random_state=5)  # oob_score(bool型)，用于配置是否使用oob样本来评估模型的泛化误差的参数。
#                                                             # 当设置为True后，在最后的模型上，即可以通过oob_score_这个属性来打印模型的oob分数。
#                                                             # oob_score_这个属性，获取的是使用OOB数据测试的R2（判定系数）分数， 
#                                                             # 也即是在oob_prediction_数据上的R2分数
# rf0.fit(X, y)

# print('oob_sore %f' % rf0.oob_score_)  # oob_sore 0.758621
# y_predprob = rf0.predict_proba(X)
# # print(y_predprob)
# y_predprob = rf0.predict_proba(X)[:, 1]
# y_pred = rf0.predict(X)
# # print(y_predprob)
# # print(y_pred)
# print('AUC Score(Train):%f' % metrics.roc_auc_score(y, y_predprob))  # AUC Score(Train):0.999951
# print('Accuracy %f' % metrics.accuracy_score(y, y_pred))  # Accuracy 0.995074
# # 此时AUC和accuracy都比较高，但是oob_score比较低， 说明过拟合 ****


# rf1 = RandomForestClassifier(n_estimators=30, oob_score=True, random_state=5) # random_state相当于seed ******
# x_test = X[-40:]
# y_test = y[-40:]
# x_train = X[:-40]
# y_train = y[:-40]

# rf1.fit(x_train, y_train)
# print('oob_score %f' % rf1.oob_score_)  # oob_score 0.773006

# y_predprob = rf1.predict_proba(x_test)[:, 1]
# y_pred = rf1.predict(x_test)
# print('AUC Score:%f' % metrics.roc_auc_score(y_test, y_predprob))  # AUC Score:0.880000
# print('Accuracy %f' % metrics.accuracy_score(y_test, y_pred))  # Accuracy 0.800000


# # 第一种方法，利用交叉验证方法对变量逐一进行调整
# def rmse_cv(model, X_train, y):
#     rmse = cross_val_score(model, X_train, y, scoring='accuracy', cv=6)   # cv表示几折交叉验证

#     return rmse

# cv_rf = rmse_cv(rf0, X, y)
# print(cv_rf)    # [ 0.73529412  0.82352941  0.76470588  0.73529412  0.91176471  0.78787879]

# NS = range(30, 100, 10)  # 表示取不同树的数目
# cv_rf1 = [rmse_cv(RandomForestClassifier(n_estimators=ns, oob_score=True, random_state=10), X, y).mean() for ns in NS]
# print(cv_rf1)  # [0.80273321449792034, 0.78297682709447403, 0.80288175876411172, 0.78787878787878796, 0.79292929292929282, 0.81253713606654776, 0.80763517528223405]

# plt.plot(NS, cv_rf1)
# plt.show()

# MD = range(5, 12, 1)
# cv_rf2 = [rmse_cv(RandomForestClassifier(n_estimators=80, max_depth=md, oob_score=True, random_state=10), X, y).mean() for md in MD]
# print(cv_rf2)

# plt.plot(MD, cv_rf2)
# plt.show()


# # 检查oob_score是否有上升
# # max_features:划分时考虑的最大特征数   min_samples_split：内部节点再划分所需最小样本数  叶子节点最少样本数
# rf3 = RandomForestClassifier(n_estimators=80, max_depth=6, max_features=5, 
#                                 min_samples_split=5, min_samples_leaf=1, oob_score=True)  
# rf3.fit(X, y)
# print('oob_score %f' % rf3.oob_score_)   # oob_score 0.798030   # 比原来的oob_score有所提高

# # 最后检查oob和AUC
# x_test = X[-40:]
# y_test = y[-40:]
# x_train = X[:-40]
# y_train = y[:-40]

# rf4 = RandomForestClassifier(n_estimators=80, max_depth=6, max_features=5, 
#                                 min_samples_split=5, min_samples_leaf=1, oob_score=True)  

# rf4.fit(x_train, y_train)
# print('oob_score %f' % rf4.oob_score_)
# y_predprob = rf4.predict_proba(x_test)[:, 1]
# print('AUC Score(test):%f' % metrics.roc_auc_score(y_test, y_predprob))
# # 进行完参数选择之后，oob_score和AUC值均有所提高，数据集小，所以最终的效果变化不大


# 下面的内容是自己写的画ROC曲线和AUC值
y_test = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
y_predprob = [0.49, 0.94, 0.56, 0.05, 0.53, 0.27, 0.9, 0.58, 0.32, 0.21]
# y_predprob=[0.75, 0.52,0.83,0.30, 0.13, 0.48,0.40,0.14,0.38, 0.30]
y_set = list(set(y_predprob))   # 为之后的截断点做前提
y_set.append(1)  # 避免了TPR，FPR取不到（0, 0）的情况  # *******
y_set.sort(reverse=True)  # 从大到小排列   #**
y_predprob = np.array(y_predprob)
xy_arr = []
for i in y_set:
    p1 = y_test[y_predprob >= i]  # 遍历y_predpred列表，当其值d >= i时， 取出y_test对应索引的值   #  *****
    p0 = y_test[y_predprob < i]
    fpR = len(p1[p1 == 0]) / (len(p1[p1 == 0]) + len(p0[p0 == 0]))   #  ****再理解
    tpR = len(p1[p1 == 1]) / (len(p1[p1 == 1]) + len(p0[p0 == 1]))
    xy_arr.append([fpR, tpR])

auc = 0.
prev_x = 0
for x, y in xy_arr:
    if x != prev_x:
        auc += (x - prev_x) * y
        prev_x = x
print('the auc is %s.' % auc)

x = [_v[0] for _v in xy_arr]
y = [_v[1] for _v in xy_arr]
plt.title('ROC curve of %s (AUC =%.4f)' % ('test', 0.87))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.plot(x, y, c='b')

plt.plot([0, 1], [0, 1], c='g')
plt.show()






























