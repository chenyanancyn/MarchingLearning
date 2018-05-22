
from math import log2
import treePlotter2
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

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


# 确定最优特征
# 1.计算全部样本基于标签的熵
def calc_shannon_ent(data, col=-1):
	N = len(data)
	label_count = {}
	for d in data:
		label_count[d[col]] = label_count.get(d[-1], 0) + 1
	shannonEnt = 0.0
	for key in label_count.keys():
		n = label_count[key]
		p = n/N
		shannonEnt -= p*log2(p)
	
	return shannonEnt


data, label = createDataSet()

# ent = calc_shannon_ent(data)
# print(ent)


# 第二步：计算利用每个特征划分后的条件熵与信息增益
# 2.1 按特征的索引和特征的值获取子集
# 2.2 把这个特征删除： 因为在建树的过程中，下一个节点分裂时，要再次计算熵，
# 这样可以减少运算量并避免该特征的影响
def split_data(data, index, value):
	sub_data = []
	for d in data:
		if d[index] == value:
			d_temp = d[:index]
			d_temp.extend(d[index+1:])
			sub_data.append(d_temp)

	return sub_data


# data, label = createDataSet()
# a = split_data(data, 0, '青年')
# print(a)


# 获取信息增益最大的特征
def get_best_split(data):
	# 求总体的熵
	base_ent = calc_shannon_ent(data)

	best_IG = 0.0
	best_index = -1

	# 进入循环
	fea_len = len(data[0]) - 1  # 特征总数
	# N = len(data[index])   # 错误
	N = len(data)  # 在计算特征对应的条件熵时， 要计算出数据的总数
	for index in range(fea_len):
		cont_ent = 0.0    # 单个特征对应的条件熵
		fea_space = set([d[index] for d in data])
		for value in fea_space:
			# 求对应特征的条件熵
			# 1.1 先找出符合条件的sub_data
			sub_data = split_data(data, index, value)
			cont_ent += len(sub_data)/N*calc_shannon_ent(sub_data)
		IG = (base_ent - cont_ent)/calc_shannon_ent(sub_data, index)   # c4.5
		if IG > best_IG:
			best_IG = IG
			best_index = index
	
	return index

# 建树
# 叶子节点
def to_leaf_node(label_list):
	label = max(set(label_list), key=label_list.count)   # ******
	return label
# 子树
# 递归建树的过程中需要维护一个特征索引列表fea_list, 每次都将最优特征索引删除
fea_list = [i for i in range(len(data[0])-1)]
def create_tree(data, fea_list):
	label_list = [d[-1] for d in data]
	if len(set(data)) == 1:   # 只剩一个数据
		return to_leaf_node(label_list)
	# if len(set(label_list)) == 1:
	if len(data[0]) == 1:    # 标签列只有一列
		return to_leaf_node(label_list)

	# 不满足截止标准继续划分
	# 获取增益最大的标签
	best_index = get_best_split(data)    # 得到最佳索引，相对索引
	
	real_index = fea_list[best_index]  # 绝对索引   *****
	del(fea_list[best_index])
	# 给树添加标签
	# 错误
	# tree['index'] = best_index
	# sub_data = 
	# tree['child'] = create_tree()
	tree = {'index': real_index, 'child': {}}
	fea_space = set([d[best_index] for d in data])
	for value in fea_space:
		sub_data = split_data(data, best_index, value)
		tree['child'][value] = create_tree(sub_data, fea_list)

	return tree



def predict(tree, sample):
	index = tree['index']
	value = sample[index]
	pre = tree['child'][value]
	if value  not in tree['child'].keys():
		return 'error'
	if isinstance(pre, dict):
		tree = pre
		pre = predict(tree, sample)
	
	return pre



		






