import numpy as np
import pandas as pd
from random import randrange


# CART解决回归问题
# 如果标签列的值不是离散的，这时使用基尼系数或者熵据无法计算不纯度，
# 因此需要新的公式计算连续标签的不纯度：最小平方误差和最小绝对误差

# 最小平方误差
def mean_square_error(groups):
    m_s_e = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        else:
            label_list = [d[-1] for d in group]
            avg = np.array(label_list).mean()    # ****
            error = np.sum(np.power(label_list - avg, 2))   # ****
        m_s_e += error

    return m_s_e


# 划分数据集
def split_data(data, index, value):
    left = []
    right = []
    for sample in data:
        if sample[index] < value:
            left.append(sample)
        else:
            right.append(sample)
    
    return left, right

# 获得最优特征和二分标准
def get_split(data, n_features):    # ++
    best_gini_score = 1e+7   # 条件基尼系数
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
            # 求取最小平方误差
            conGini = mean_square_error(groups)

            if conGini < best_gini_score:
                best_gini_score = conGini
                best_index = feature
                best_value = feature_value
                best_groups = groups

    return best_index, best_value, best_groups, best_gini_score  # 增加了输出最小误差  ++


# 停止条件和变成叶子节点
# 分类预测时，其中一个停止条件是：如果全部的标签都是一样的，那么返回
# 如果是连续标签，标签值全部相同几乎不可能设定最小平方误差阈值stop_mse代替以上停止条件
def to_leaf_node(label_list):
    return np.array(label_list).mean()


# 递归建树
def create_tree(data, n_features, max_depth, stop_mse, min_size, depth=0):   # ++
    label_list = [d[-1] for d in data]
    label_space = list(set(label_list))
    n_features = len(data[0]) - 1
    if len(label_space) <= min_size:    # 这个判断加上，因为很有可能标签种类会小于min_size
        min_size = 1
    if (len(label_space) <= min_size) or (len(label_space) == 1) or depth == max_depth:
        return to_leaf_node(label_list)
    if stop_mse <= 1e-7:    # 增加了最小误差停止条件 +++
        return to_leaf_node(label_list)
    else:
        best_index, best_value, groups, stop_mse = get_split(data, n_features)  # ++
        # 错误
        # tree['index'] = best_index
        # tree['value'] = best_value
        # tree['groups'] = groups    
        tree = {'index': best_index, 'value': best_value, 'left': {}, 'right': {}} 
        tree['left'] = create_tree(groups[0], n_features, max_depth, stop_mse, min_size, depth+1)    # 数据的放入
        tree['right'] = create_tree(groups[1], n_features,  max_depth, stop_mse, min_size, depth+1) 
        return tree


def fit(data, stop_mse, n_features=None, max_depth=None, min_size=1):  # +++
    if max_depth is None:   # ******
        max_depth = 100
    return create_tree(data, n_features,  max_depth, stop_mse, min_size)  # ++


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


# 计算预测的均方误差
def m_s_e(tree, test_data, test_label):
    mse = 0.0
    pre_label = [predict(tree, d) for d in test_data]
    for index in range(len(test_label)):
        d_value = (test_label[index] - pre_label[index]) ** 2
        mse += d_value
    
    return mse


# 建树与预测
winequality_red_df = pd.read_csv('winequality-red.csv', index_col=None)   
# print(winequality_red_df.head())   # 此时数据为以；隔开的string
# print(winequality_red_df)
datas = winequality_red_df.values   # 会根据数据的类型自动转换     
# print(type(data))    # numpy
# print(data)
data = []   # 数据集
for d in datas:
    temp = [float(da) for da in (d[0].split(';'))]
    data.append(temp)
data = np.array(data)
# print(data)
# 将数据打乱
np.random.shuffle(data)   # 返回值为None
# print(data)
# # 将数据集分为训练和验证两部分
# print(data.shape)   # 1599, 12
train = data[:1280]
# print(train.shape)
validation = data[1280:]
validation_label = [d[-1] for d in validation]
validation_data = [d[:-1] for d in validation]
# 训练并建树
tree = fit(train, max_depth=4, min_size=2, stop_mse=1e+7)    # 当max_depth适度变小时，正确率会提高 ****   
print(tree)
# 计算均方误差
m_s_e = m_s_e(tree, validation_data, validation_label)
print(m_s_e)