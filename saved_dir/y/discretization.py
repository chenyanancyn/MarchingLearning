
'''将连续的特征进行离散化，默认是10份'''
import numpy as np


def feature_discretization(dataSet, method="step", step_num=10):
	data_set_copy = dataSet.copy()   # 深度复制，指针的位置不同
	m = len(data_set_copy)    # 共有多少个数据
	n = len(data_set_copy[0])   # 数据有多少列
	for i in range(n-1):    # 因为数据的最后一列为标签，所以-1
		unique_num = set([row[i] for row in data_set_copy])   # 每个特征的值
		min_num, max_num = min(unique_num), max(unique_num)   # 每个特征的最小、最大值
		step = (max_num-min_num)/step_num     # 每个特征的步长 
		if max_num-min_num == 0:
			continue
		else:
			for j in range(m):   # 对每个数据进行循环
				data_set_copy[j][i] = int((data_set_copy[j][i]-min_num)/step)  # 将离散数值分到range(10)

	return data_set_copy
