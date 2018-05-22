
import pandas as pd

import numpy as np

###mse 此时不能保证是小于1的    # ++ 为啥？

dataSet=[[1,1,'yes'],
	[1,1,'yes'],
	[1,0,'no'],
	[0,1,'no'],
	[0,1,'no']]

def split_data(data,index,value):
	left=[];right=[]
	for d in data:
		if (d[index]<value):
			left.append(d)
		else:
			right.append(d)
	return left, right

def mean_squar_error(groups):
	m_s_e=0.0
	for group in groups:
		if len(group)==0:
			continue
		label_list=[d[-1] for d in group]
		c=np.array(label_list).mean()
		m_s_e+=np.sum(np.power(label_list-c,2))
	return m_s_e


def get_split(data):
	fea_num=len(data[0])-1
	min_gini=999;
	b_index=-1;b_value=-1
	b_groups=()
	for i in range(fea_num):
		fea_space=set([d[i] for d in data])
		for value in fea_space:
			groups=split_data(data, i, value)
			cond_gini=mean_squar_error(groups)
			if cond_gini<min_gini:
				min_gini=cond_gini
				b_index=i;b_value=value
				b_groups=groups
	return b_index,b_value,b_groups


def to_leaf_node(label_list):

	return np.mean(label_list)

def create_tree(data,max_depth,min_size,depth=0):
	label_list=[d[-1] for d in data]
	if (len(data)<=min_size) or (len(set(label_list))==1) or (depth>=max_depth):		
		return to_leaf_node(label_list)

	index,value,groups=get_split(data)
	if(mean_squar_error(groups)<1e-4):
		return to_leaf_node(label_list)
	tree={'index':index,'value':value,'left':{},'right':{}}

	tree['left']=create_tree(groups[0], max_depth,min_size,depth+1)
	tree['right']=create_tree(groups[1], max_depth,min_size,depth+1)
	
	return tree

def fit(data,max_depth=None,min_size=1):
	if max_depth==None:
		max_depth=100
	return create_tree(data, max_depth, min_size)

def predict(tree,sample):
	index=tree['index'];value=tree['value']
	if sample[index]<value:
		if isinstance(tree['left'],dict):
			 return predict(tree['left'], sample)
		else:
			 return tree['left']
	else:
		if isinstance(tree['right'],dict):
			 return predict(tree['right'], sample)
		else:
			 return tree['right']


import treePlotter_cart
data=pd.read_csv('sonar.csv',header=None)
print(data.head())
data[data.columns[-1]]=data[60].map({'M':0,'R':1})
print(data.head())
train_data=data.values.tolist()
np.random.shuffle(train_data)
tree=fit(train_data)
print(tree)
treePlotter_cart.createPlot(tree)







