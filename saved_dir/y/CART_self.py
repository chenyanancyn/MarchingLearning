import pandas as pd
import treePlotter2
from pylab import *
import numpy as np
from random import randrange
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


# dataSet=[[1,1,'yes'],
# 	[1,1,'yes'],
# 	[1,0,'no'],
# 	[0,1,'no'],
# 	[0,1,'no']]


# 选择最优特征与二分标准
# 利用（特征、二分标准）对样本集进行分割
# 计算条件基尼系数
# 获得最优的特征与最优二分标准

# 决策点过程：
# 1.对每个特征，以及可能的二分标准，将数据集分为左右两个子数据集
# 2.计算分割的左右子数据集的条件基尼系数，在所有分割情况中，基尼系数最小的特征
# 与二分标准即为最优切分点

# 分割数据集
def split_data(data, index, value):
    left = []
    right = []
    for sample in data:
        if sample[index] < value:
            left.append(sample)
        else:
            right.append(sample)
    
    return left, right


# 基尼系数
# 1.求基尼系数
# 2.求条件基尼系数
# 3.合并
# 1.样本总体的基尼系数gini
# def gini(data):
#     N = len(data)
#     label_list = [d[-1] for d in data]  # 将标签的所有取值都取到
#     label_space = set(label_list)
#     giniScore = 0.0
#     for label in label_space:
#         n = label_list.count(label)
#         giniScore += n/N * (1-n/N)

#     return giniScore

# # 2.计算条件基尼系数condition_gini
# def condition_gini(groups):
#     N = len(groups[0] + groups[1])
#     conGini = 0.0
#     for group in groups:
#         giniScore = gini(group)
#         conGini += giniScore * len(group) / N

#     return conGini


# 3.合并
def condition_gini(groups):
    N = len(groups[0]) + len(groups[1])
    conGini = 0.0
    for group in groups:
        gini = 0.0
        n = len(group)
        label_list = [d[-1] for d in group]
        label_space = {}
        for label in label_list:
            label_space[label] = label_space.get(label, 0) + 1
        for value in label_space.values():
            gini += value/n * (1 - value/n)
        conGini += gini * n / N
    
    return conGini


# 最优特征与最优二分标准
def get_split(data, n_features):    # ++
    best_gini_score = 1   # 条件基尼系数
    # best_index = -1
    # best_value = 0.0    # 不用提前设出来，因为在后面没有比较的需求
    # feature_list = len(data[0]) - 1   # ini
    feature_list = []   # ++ 
    N = len(data[0]) - 1   # ++ 
    if n_features == N:  # ++
        feature_list = range(N)   # ++ 
    else:   # ++ 
        for i in range(n_features):   # ++ 
            index = randrange(N)   # ++ 
            feature_list.append(index)   # ++
    for feature in feature_list:  # ++
        feature_values = list(set([d[feature] for d in data]))   # 找出一个特征中的所有值
        for feature_value in feature_values:
            # 分割数据集
            groups = split_data(data, feature, feature_value)
            # 求取条件基尼系数
            conGini = condition_gini(groups)

            if conGini < best_gini_score:
                best_gini_score = conGini
                best_index = feature
                best_value = feature_value
                best_groups = groups

    return best_index, best_value, best_groups


# print(get_split(dataSet))


# 递归建树
# 添加create_tree(data, max_depth, min_size, depth)函数考虑三个问题：
# 1.终止条件 
# 2.递归过程
# 3.节点结构

# 1.终止条件
# 与ID3不同，此时的特征不会减少
# 如果样本标签全部相同停止
# 如果样本的数量少于给定阈值（3或5）就停止
# 如果树的深度大于给定阈值（8或10）就停止
# 停止的意思是从决策节点变成叶子节点并跳出
# 因此也需要一个to_leafe_node函数
# 输入决策点的所有标签，返回出现次数最多的标签


# 节点结构
# 1.使用字典来保存
# 2.保存决策点的特征索引、二分标准、左子树、右子树、左右子样本集
# 注意：get_split函数返回的groups是左右样本集，并不是左右子树
# tree = {'index': index, 'value': value, 'left': left, 'right': right, 'groups': groups}

# # 建树步骤

# 叶子节点   *****
def to_leaf_node(label_list):
    return max(label_list, key=label_list.count)


def create_tree(data, n_features, max_depth, min_size, depth=0):   # ++
    label_list = [d[-1] for d in data]
    label_space = list(set(label_list))
    n_features = len(data[0]) - 1
    if len(label_space) <= min_size:    # 这个判断加上，因为很有可能标签种类会小于min_size
        min_size = 1
    if (len(label_space) <= min_size) or (len(label_space) == 1) or depth == max_depth:
        return to_leaf_node(label_list)
    else:
        best_index, best_value, groups = get_split(data, n_features)  # ++
        # 错误
        # tree['index'] = best_index
        # tree['value'] = best_value
        # tree['groups'] = groups    
        tree = {'index': best_index, 'value': best_value, 'left': {}, 'right': {}} 
        tree['left'] = create_tree(groups[0], n_features, max_depth, min_size, depth+1)    # 数据的放入
        tree['right'] = create_tree(groups[1], n_features, max_depth, min_size, depth+1) 
        tree['groups'] = groups
        return tree


def fit(data, n_features=None, max_depth=None, min_size=1):  # +++
    if max_depth is None:   # ******
        max_depth = 100
    return create_tree(data, n_features, max_depth, min_size)  # ++

dataSet = [[1, 2, 'yes'], [1, 2, 'yes'], [1, 0, 'no'], 
            [1, 1, 'no'], [0, 2, 'no']]

# fresh_tree = fit(dataSet)
# print(fresh_tree)

# treePlotter2.createPlot(fresh_tree)   # ++


# CART分类预测
def predict(tree, sample):
    index = tree['index']
    value = tree['value']
    if sample[index] < value:
        if isinstance(tree['left'], dict):
            return predict(tree['left'], sample)    
        else:
            return tree['left']
    else:
        if isinstance(tree['right'], dict):
            return predict(tree['right'], sample)    
        else:
            return tree['right']


# sonar_df = pd.read_csv('sonar.csv')

# sonar_df = pd.read_csv('sonar.csv', header=None)
# sonar_df = pd.read_csv('sonar.csv', header=None, prefix='V')    
# print(sonar_df.head())
# # print(sonar_df.V60.groupby())
# data=sonar_df.values.tolist()
# # print(data)
# # node=get_split(data)
# # # print(node[0],node[1])
# # print(node[2][0][0][10])


# 建树与预测
sonar_df = pd.read_csv('sonar.csv', header=None, index_col=None)
data = sonar_df.values   # 会根据数据的类型自动转换     
# print(type(data))    # numpy
# # 将数据打乱
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
tree = fit(train, max_depth=10, min_size=2)    # 当max_depth适度变小时，正确率会提高 ****   
# print(tree)
# 进行预测
# 剪枝前
pre = [predict(tree, d) for d in validation_data]
# 计算正确率
# accuracy = (pre == validation_label).tolist().count(True)/len(validation_label)   
count = 0
for i in range(len(pre)):
    if pre[i] == validation_label[i]:
        count += 1
# print(count)
# print(len(pre))
accuracy = count/len(validation_label)
print(accuracy)

# 进行剪枝
from  cart_purning_self import purning_CART
tree_pruning = purning_CART(tree, validation_data)
# 剪枝后预测
pre = [predict(tree_pruning, d) for d in validation_data]
# 计算正确率
# accuracy = (pre == validation_label).tolist().count(True)/len(validation_label)   
count = 0
for i in range(len(pre)):
    if pre[i] == validation_label[i]:
        count += 1
# print(count)
# print(len(pre))
accuracy = count/len(validation_label)
print(accuracy)
