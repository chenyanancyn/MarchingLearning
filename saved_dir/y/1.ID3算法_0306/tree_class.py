from math import log2
import treePlotter2
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
import pandas as pd  # 导入包，在读取数据时使用

# treePlotter2.createPlot(myTree)


def createDataSet():
	dataSet = [['青年', '否', '否', '一般', '拒绝'],
				['青年', '否', '否', '好', '拒绝'],
				['青年', '是', '否', '好', '同意'],
				['青年', '是', '是', '一般', '同意'],
				['青年', '否', '否', '一般', '拒绝'],
				['中年', '否', '否', '一般', '拒绝'],
				['中年', '否', '否', '好', '拒绝'],
				['中年', '是', '是', '好', '同意'],
				['中年', '否', '是', '非常好', '同意'],
				['中年', '否', '是', '非常好', '同意'],
				['老年', '否', '是', '非常好', '同意'],
				['老年', '否', '是', '好', '同意'],
				['老年', '是', '否', '好', '同意'],
				['老年', '是', '否', '非常好', '同意'],
				['老年', '否', '否', '一般', '拒绝'],
				]
	# feature_index = ['年龄', '有工作', '有房子', '信贷情况']
	feature_index = [0, 1, 2, 3]
	return dataSet, feature_index   # 要维护索引列表

def calc_shannon_ent(data, col=-1):
	label_list = [d[col] for d in data]
	label_count = {}
	for label in label_list:
		# label_count[label] = label_count.get(label, 0) + 1
		if label not in label_count.keys():
			label_count[label] = 0
		label_count[label] += 1

	# 按照公式求标签信息熵
	N = len(data)
	ent = 0.0
	for value in label_count.values():
		ent -= value/N*log2(value/N)
	return ent


# data, label = createDataSet()

# ent = calc_shannon_ent(data)
# print(ent)

def split_data(data, index, value):
	sub_data = []
	for data_i in data:
		if data_i[index] == value:
			# del (data_i[index])
			d=data_i[:index];d.extend(data_i[index+1:])
			sub_data.append(d)

	return sub_data


# data, label = createDataSet()
# a = split_data(data, 0, '青年')
# print(a)


def choose_best_index(data):
	# 求总体的熵
	base_ent = calc_shannon_ent(data)
	best_IG = 0.0
	best_index = -1
	# 确定样本即有多少个特征
	fea_len = len(data[0]) - 1  #  因为最后一个是标签
	N = len(data)
	for index in range(fea_len):
		cont_ent = 0.0
		fea_space = set([d[index] for d in data])
		for value in fea_space:
			sub_data = split_data(data, index, value)
			cont_ent += calc_shannon_ent(sub_data)*len(sub_data)/N
		IG = (base_ent - cont_ent)/calc_shannon_ent(data, index)   # c4.5
		if IG > best_IG:
			best_IG = IG
			best_index = index
	
	return best_index

# data, label = createDataSet()
# a = choose_best_index(data)
# print(a)


def to_leaf_node(label_list):
	label = max(set(label_list), key=label_list.count())   # 取出出现次数最多的标签
	return label


fea_list = [i for i in range(len(data[0])-1)]   # 要维护的索引列表
def create_tree(data, fea_list):
	label_list = [d[-1] for d in data]
	# if label_list.count(label_list) == len(label_list):   # 与下面判断条件效果相同
	if len(set(label_list)) == 1:
		# return to_leaf_node(label_list)
		return label_list[0]
	if len(data[0]) == 1:  # 表只剩标签列即：所有特征用完
		return to_leaf_node(label_list)
	# if len(set(label_list)) == 1 or len(data[0]) == 1:
	# 	return to_leaf_node(label_list)

	# 不满足截至标准进行划分
	index = choose_best_index(data)   # 得到最佳索引（相对索引）

	real_index = fea_list[index]  # 绝对索引
	del(fea_list[index])

	tree = {'index': real_index, 'child': {}}
	fea_space = set([d[index] for d in data])
	for value in fea_space:
		sub_data = split_data(data, index, value)
		tree['child'][value] = create_tree(sub_data, fea_list)

	return tree

# tree = create_tree(data, fea_list)
# print(tree)
# treePlotter2.createPlot(tree)

def predict(tree, sample):
	index = tree['index']
	value = sample[index]
	pre = tree['child'][value]
	if value not in tree['child'].keys():
		return 'error'
	if isinstance(pre, dict):
		tree = pre
		pre = predict(tree, sample)
	return pre


# pre = predict(tree, ['青年', '否', '否', '一般', '拒绝'])
# print(pre)


import discretization
# data = read_csv('../sonar.csv', header=None, index_col=0)   # 第二个默认有行标,所以要标None,第三个默认为None,所以可以不写 
# print(data.head())    # 加上head就是默认输出5行数据

# data = data.values.tolist()
# data = discretization.feature_discretization(data)

# fea_list = [i for i in range(len(data[0]-1))]
node = get_split(data)
print(node[0], node[1])   # 第一个是划分标准即特征， 第二个是划分值
print(node[2][0][0][10])
# tree = crea4(data, fea_list)

# print(tree)
# treePlotter2.createPlot(tree)
