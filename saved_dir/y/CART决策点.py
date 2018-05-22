
import pandas as pd
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

# def gini(data):
# 	label_list=[d[-1] for d in data]
# 	label_space=set(label_list)
# 	N=len(label_list)
# 	gini=0.0
# 	for label in label_space:
# 		pi=label_list.count(label)/N 
# 		gini+=pi*(1-pi)
# 	return gini

# def condition_gini(groups):
# 	N=len(groups[0])+len(groups[1])
# 	cond_gini=0.0
# 	for data in groups:
# 		gini=gini(data)
# 		cond_gini+=len(data)/N*gini
# 	return cond_gini

def condition_gini(groups):
	N=len(groups[0])+len(groups[1])
	cond_gini=0.0
	for data in groups:
		if(len(data)==0):
			continue
		label_list=[d[-1] for d in data]
		label_space=set(label_list)
		for label in label_space:
			pi=label_list.count(label)/len(data)
			cond_gini+=len(data)/N*pi*(1-pi)
	return cond_gini


def get_split(data):
	fea_num=len(data[0])-1
	min_gini=1;
	b_index=-1;b_value=-1
	b_groups=()
	for i in range(fea_num):
		fea_space=set([d[i] for d in data])
		for value in fea_space:
			groups=split_data(data, i, value)
			cond_gini=condition_gini(groups)
			if cond_gini<min_gini:
				min_gini=cond_gini
				b_index=i;b_value=value
				b_groups=groups
	return b_index,b_value,b_groups

# left,right=split_data(dataSet,0,0)

# cond_gini=condition_gini((left,right))
# print(cond_gini)

# node=get_split(dataSet)
# print(node)

data_df=pd.read_csv('sonar.csv',header=None)
print(data_df.head())
data=data_df.values.tolist()

node=get_split(data)
print(node[0],node[1])
print(node[2][0][0][10])

