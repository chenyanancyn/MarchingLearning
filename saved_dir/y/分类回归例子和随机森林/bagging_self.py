# 集成算法
# 集成算法： 并行方法：Bageing、随机森林RF(Random Forest)
#          串行方法(Boosting族)：Adaboost, GBM

# 1.集成算法
# 如何选择、生成“个体模型”：若学习模型：决策树、决策树桩
# 如何综合多个模型：将相同类型但训练集不同的弱分类器进行提升
#                将相同类型但权重不同的若分类器进行提升
#                将不同类型的弱分类器进行提升

# 集成框架的两种模式：
# 第一种：期望个个分类器之间依赖性不强，可以同时进行生成。这种做法称为并行方法，代表为Bagging,适用性更强的拓展是随机森林
#       并行集成方法的重点：如何从总体训练集获得多个不同的子训练集
# 第二种：弱学习器之间具有强依赖性，只能序列生产，称之为串行方法。代表方法有：Boosting,包括Adaboost,GBDT,XGBoost和 lightGBM
#       串行集成方法的重点：如何更新弱分类器的权重（弱分类器的整体“话语权”，以及某一样本的识别效果）

# 2.Bagging和随机森林
# Bagging是于1996年的Breiman提出的，他的思想根源是数理统计学中的非常重要的Bootstrap理论
# Bootstrap：‘自举’（自采样），通过模拟的方法来逼近样本的概率分布      子样本之于样本，类比于样本之于总体


# bootstrap的实现
import random
def bootstrap(data):
    N = len(data)
    sub_data = []
    for i in range(N):
        index = random.randrange(N)    # ******
        sub_data.append(data[index])

    return sub_data


# Bagging方法
# Bagging方法：全称Bootstrap Aggregating
# 用Bootstrap生成M个数据集
# 用这M个数据集训练出M个弱分类器
# 最终模型即为这M个弱分类器的简单组合

# 所谓简单组合就是： 对于分类问题使用投票表决的方法   对于回归问题使用简单的取平均
# Bagging算法实现
from CART_self import fit, predict
def bagging(data_set, n_trees=5):
    trees = []
    for i in range(n_trees):
        sub_samples = bootstrap(data_set)
        tree = fit(sub_samples, max_depth=5)
        trees.append(tree)

    return trees

# Bagging算法预测
# 使用trees和一个测试样本的特征，预测该样本的类别bagging_predict

def bagging_predict(trees, sample):
    b_p = []
    for tree in trees:
        b_p.append(predict(tree, sample))

    return max(b_p, key=b_p.count)



# 建树与预测
import pandas as pd
import numpy as np
sonar_df = pd.read_csv('sonar.csv', header=None, index_col=None)
data = sonar_df.values   # 会根据数据的类型自动转换     
# print(type(data))    # numpy
# 将数据打乱
np.random.shuffle(data)   # 返回值为None
# print(data)
# 将数据集分为训练和验证两部分
# print(data.shape)   208, 61
train = data[:180]
# print(train)
validation = data[180:]
validation_label = [d[-1] for d in validation]
validation_data = [d[:-1] for d in validation]
# 训练并建树
# tree = fit(train, max_depth=10, min_size=2)    # 当max_depth适度变小时，正确率会提高 ****   ++为啥
trees = bagging(train)
# print(tree)
# 进行预测
# pre = [predict(tree, d) for d in validation_data]
pre = [bagging_predict(trees, d) for d in validation_data]
# 计算正确率
# accuracy = (pre == validation_label).tolist().count(True)/len(validation_label)   # ++ 
count = 0
for i in range(len(pre)):
    if pre[i] == validation_label[i]:
        count += 1
# print(count)
# print(len(pre))
accuracy = count/len(validation_label)
print(accuracy)