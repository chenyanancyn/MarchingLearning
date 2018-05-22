
import numpy as np

data=[[1,2,3],[2,3,4],[4,5,6]]
label=[[1],[1],[-1]]

def split_data(data,index,value,lr):
	pred_label=[]
	if lr=='l':
		for d in data:
			if d[index]<value:
				pred_label.append(1)
			else:
				pred_label.append(-1)
	if lr=='r':
		for d in data:
			if d[index]<value:
				pred_label.append(-1)
			else:
				pred_label.append(1)
	return pred_label	
# print(split_data(data,1,3,'l'))

def split_data(data,index,value,lr):
	data=np.array(data)
	m,n=data.shape
	pred_label=np.ones((m,1))

	if lr=='l':
		pred_label[data[:,index]>=value]=-1
	if lr=='r':
		pred_label[data[:,index]<value]=-1
	return pred_label


# print(split_data(data,1,3,'r'))

def tree_stump(data,label,w):
	data=np.array(data)

	m,n=data.shape
	label=label.reshape((m,1))
	w=w.reshape((m,1))
	min_w_err=999
	b_pred=np.zeros((m,1))
	stump={'index':-1,'value':-1,'lr':'l'}

	for index in range(n):
		fea=set(data[:,index])
		for value in fea:
			for lr in ['l','r']:
				pred_label=split_data\
				(data, index, value, lr)
				err=np.zeros((m,1))
				err[pred_label!=label]=1
				w_err=np.matmul(err.T,w)
				if w_err<min_w_err:
					min_w_err=w_err
					b_pred=pred_label
					stump['index']=index
					stump['value']=value
					stump['lr']=lr
	return min_w_err , b_pred , stump




def adaboost(data,label,n_stump):
	data=np.array(data)
	m,n=np.shape(data)#m.n=data.shape
	label=label.reshape((m,1))
	boost_stump={}
	w=np.ones((m,1))/m
	pred_label=np.zeros((m,1))
	for i in range (n_stump):
		w_err,b_pred,stump=tree_stump(data, label, w)
		alpha=0.5*np.log((1-w_err)/max(w_err,1e-8))
		e_ayg=np.exp(-alpha*np.multiply(label,b_pred))
		w=np.multiply(w,e_ayg)
		w=w/np.sum(w)
		boost_stump[i]=(alpha,stump)
		######算法流程完成，检测当前的效果
		pred_label+=b_pred*alpha
		label_current=np.ones((m,1))
		label_current[pred_label<0]=-1
		print((label_current==label).flatten().tolist().count(True)/m)
		print(alpha,stump)
		#####算法流程完成，检测当前的效果
	return boost_stump



# stump={'index':0,'value':2,'lr':'l'}
# x=np.array([3,2,1])
def stump_predict(stump,x):
	index=stump['index']
	value=stump['value']
	lr=stump['lr']
	if x[index]<value:
		y_pred =1 if lr=='l' else -1
	else:
		y_pred =-1 if lr=='l' else 1
	return y_pred


def adaboost_predict(boost_stump,x):
	pred_label=0
	M=len(boost_stump.keys())
	for i in range(M):
		alpha,stump=boost_stump[i]
		y_pred=stump_predict(stump, x)
		pred_label+=alpha*y_pred
	if pred_label>0:
		return 1
	else: return -1


# data=np.array([[1,2,3],[2,3,4],[4,5,6]])
# label=np.array([[1],[-1],[1]])

# adaboost_stump=adaboost(data,label,3)
# print(adaboost_stump)
# print(adaboost_predict(adaboost_stump,np.array([2,3,4])))

import pandas as pd
data_df=pd.read_csv('sonar.csv',header=None)

data_df[60]=data_df[60].map({'M':1,'R':-1})
# print(data_df.head(5))

data_array=data_df.values

data=data_array[:,:-1]
label=data_array[:,-1]
print(label.shape)

adaboost_stump=adaboost(data,label,20)
