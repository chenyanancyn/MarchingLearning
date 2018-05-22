import pandas as pd


dataSet=[[1,1,'yes'],
	[1,1,'yes'],
	[1,0,'no'],
	[0,1,'no'],
	[0,1,'no']]


def split_data(data, index, value):
	left = []
	right = []
	for d in data:
		if (d[index] < value):
			left.append(d)
		else:
			right.append(d)
	return left, right


# # 基尼系数
# def gini(data):  # 参照求熵的函数
# 	label_list = [d[-1] for d in data]
# 	label_space = set(label_list)   # 找到列表中所有取值的情况
# 	N = len(label_list)
# 	gini = 0.0
# 	for label in label_space:
# 		pi = label_list.count(label)/N
# 		gini += pi*(1-pi)
# 	return gini


# # 计算条件基尼系数
# def condition_gini(groups):
# 	N = len(groups[0]) + len(groups[1])
# 	cond_gini = 0.0
# 	for data in groups:
# 		gini = gini(data)
# 		cond_gini += len(data)/N*gini
# 	return cond_gini

def condition_gini(groips):
	N = len(groups[0]) + len(groups[1])
	cond_gini = 0.0
	for data in groups:
		label_list = [d[-1] for d in data]
		label_space = set(label_list)
		for label in label_space:
			pi = label_list.count(label)/len(data)
			cond_gini += len(data)/N*pi*(1-pi)
	return cond_gini


def get_split(data):
	fea_num = len(data[0]) - 1
	min_gini = 1  # 条件基尼系数最大值为1
	for i in range(fea_num):
		fea_space = set([d[i] for d in data])
		for value in fea_space:
			groups = split_data(data, i, value)
			cond_gini = condition_gini(groups)
			if cond_gini < min_gini:
				min_gini == cond_gini
				b_index = i
				b_value = value
				b_groups = groups
	return b_index, b_value, b_groups


# left, right = split_data(dataSet, 0, 0)

# cond_gini = condition_gini((left, right))
# print(cond_gini)

# node = get_split(dataSet)
# print(node)


# CART建树
def to_leaf_node(label_list):
    return max(set(label_list), key=label_list.count)


def create_tree(data, max_depth, min_size, depth=0):
    label_list = [d[-1] for d in data]
    if (len(data) <= min_size) or len(set(label_list)) == 1 or (depth == max_depth):
        return to_leaf_node(label_list)
    
    index, value, groups = get_split(data)
    tree = {'index': index, 'value': value, 'left': {}, 'right': {}}
    tree['left'] = create_tree(groups[0], max_depth, min_size, depth+1)
    tree['right'] = create_tree(groups[1], max_depth, min_size, depth+1)
    tree['groups'] = groups
    return tree

def fit(data, max_depth=None, min_size=1):
    if max_depth == None:
        max_depth = 100
    return create_tree(data, max_depth, min_size)   # 在这个函数中给出递归的默认的值，避免出错 3


data = pd.read_csv('sonar.csv', header=None)
tree = create_tree(data,)
print(tree)

# import treePlotter_cart
# treePlotter_cart.create_Plot(tree)

# 预测树
def predict(tree, sample):
    index = tree['index']
    value = tree['value']
    if example[index] >= value:    # 右树
        if isinstance(tree['right'], dict):
            return predict(tree['right'], sample)
        else:
            return tree['right']
    else:                          # 左树
        if isinstance(tree['left'], dict):
            return predict(tree['left'], sample)
        else:
            return tree['left']


# 最小平方误差的实现
def mean_square_error(groups):
    m_s_c = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        labels = [row[-1] for row in group]
        proportion = np.array(labels).mean()
        error = np.sum(np.power(labels - proportion, 2))
        m_s_c += error

    return m_s_c


import pandas as pd
import numpy as np
import CART_Classifier
import treePlotter_cart
sonar_df = pd.read_csv('sonar.csv', header=None, prefix='V')

print(sonar_df.head())
print(sonar_df.V60.groupby(sonar_df))


import pandas as pd
abalone=pd.read_csv('abalone.csv', header=None, prefix='V')









        

