
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
	feature_index=[0,1,2,3]
	return dataSet,feature_index


def calc_shannon_ent(data):
	label_list=[d[-1] for d in data]
	label_count={}
	for label in label_list:
		label_count[label]=label_count.get(label,0)+1
		# if label not in label_count.keys():
		# 	label_count[label]=0
		# label_count[label]+=1
	N=len(data)
	ent=0.0
	for value in label_count.values():
		ent-=value/N*log2(value/N)
	return ent
data,label=createDataSet()

ent=calc_shannon_ent(data)
print(ent)

def split_data(data,index,value):
	sub_data=[]
	for d in data:
		if d[index]==value:
			sub_data.append(d)
	return sub_data

def choose_best_index(data):
	base_ent=calc_shannon_ent(data)
	best_IG=0.0;best_index=-1
	fea_len=len(data[0])-1
	N=len(data)
	for index in range(fea_len):
		cond_ent=0.0
		fea_space=set([d[index] for d in data])
		for value in fea_space:
			sub_data=split_data(data, index, value)
			cond_ent+=calc_shannon_ent(sub_data)*len(sub_data)/N
		IG=base_ent-cond_ent
		if IG>best_IG:
			best_IG=IG;best_index=index
	return best_index
print(choose_best_index(data))

def to_leaf_node(label_list):
	label=max(set(label_list),key=label_list.count())
	return label


fea_list=[i for i in range(len(data[0])-1)]
def create_tree(data,fea_list):
	label_list=[d[-1] for d in data]
	# if label_list.count(label_list[0])==len(label_list):
	if len(set(label_list))==1:
		return label_list[0]
	if len(data[0])==1:
		return to_leaf_node(label_list)

	# if len(set(label_list))==1 or len(data[0])==1:
	# 	return to_leaf_node(label_list)
	index=choose_best_index(data)

	real_index=fea_list[index]
	del(fea_list[index])

	tree={'index':real_index,'child':{}}
	fea_space=set([d[index] for d in data])
	for value in fea_space:
		sub_data=split_data(data, index, value)
		tree['child'][value]=create_tree(sub_data, fea_list)
	return tree
tree=create_tree(data, fea_list)
print(tree)
treePlotter2.createPlot(tree)