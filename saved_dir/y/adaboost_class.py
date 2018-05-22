import numpy as np


# def split_data(data, index, value, lr):
#     pred_label = []
#     if lr == 'l':
#         for d in data:
#             if d[index] < value:
#                 pred_label.append(1)
#             else:
#                 pred_label.append(-1)
#     else:
#         for d in data:
#             if d[index] < value:
#                 pred_label.append(-1)
#             else:
#                 pred_label.append(1)

#     return pred_label


def split_data(data, index, value, lr):
    data = np.array(data)
    m, n = data.shape
    pred_label = np.ones((m, 1))
    if lr == 'l':
        pred_label[data[:, index] > value] = -1
    if lr == 'r':
        pred_label[data[:, index] < value] = -1
    
    return pred_label

data = [[1, 2, 3], [3, 4, 5], [4, 5, 6]]

# print(split_data(data, 1, 3, 'l'))

def tree_stump(data, labelList, w):
    data = np.array(data)
    m, n = data.shape
    w = w.reshape((m, 1))
    labelList = labelList.reshape((m, 1))
    min_x_err = 999
    b_pred = np.zeros((m, 1))
    stump = {'index': -1, 'value': -1, 'lr': '1'}
    for index in range(n):
        fea = set(data[:, index])
        for value in fea:
            for lr in ['l', 'r']:
                pre_label = split_data(data, index, value, lr)
                err = np.zeros((m, 1))
                err[pre_label != labelList] = 1
                w_err = np.matmul(err.T, w)    # 求内积   # np.multiply      # array对应项相乘
                if w_err < min_x_err:
                    min_x_err = w_err
                    b_pred = pred_label
                    stump['index'] = index
                    stump['value'] = value
                    stump['lr'] = lr

    return min_x_err, b_pred, stump


def adaboost(data, label,  n_stump):
    data = np.array(data)
    m, n = np.shape(data)
    label = label.reshape((m, 1))

    boost_stump = {}
    w = np.ones((m, 1)) / m

    for n in range(n_stump):
        w_err, b_pred, stump = tree_stump(data, label, w)
        alpha = 0.5 * np.log((1 - w_err)/max(w_err, 1e-8))

        e_ayg = np.exp(-alpha * np.multiply(label, b_pred))
        w = np.multiply(w, e_ayg)
        pred_label  = np.zeros((m ,1))

        w = w / np.sum(w)
        # Z = np.multiply(w, e_ayg)
        # w = w/Z
 
        boost_stump[n] = (alpha, stump)

        ### 算法流程完成，检测当前效果
        # pred_label += b_pred * alpha
        # label_current = np.ones((m, 1))
        # label_current[label_current < 0] = -1
        # print((label_current == label).flatten().tolist().count(True)/m)
        # print(w)
        ### 算法流程完成，检测当前效果

    return boost_stump

# data = np.array([[1,2,3],[2,3,4],[4,5,6]])
# label = np.array([[1], [-1], [1]])

# adaboost_stump = (data, label, 3)

# def adaboost_predict():
stump = {'index':0, 'value':2, 'lr':'l'}
x = np.array([3,2,1])
def stump_predict(stump, x):
    index = stump['index']
    value = stump['value']
    lr = stump['lr']
    if x[index] < value:
        # if 'lr' == 'l':
        #     y_pred = 1
        # else:
        #     y_pred = -1
        y_pred = 1 if lr == 'l' else -1
    else:
        # if 'lr' == 'l':
        #     y_pred = -1
        # else:
        #     y_pred = 1
        y_pred = -1 if lr == 'l' else 1

    return y_pred

print(stump_predict(stump, x))

def adaboost_predict(boost_stump, x):
    pred_label = 0
    M = len(boost_stump.keys())
    for i in range(M):
        alpha, stump = boost_stump[i]
        y_pred = stump_predict(stump, x)
        pred_label += alpha * y_pred

    if pred_label > 0:
        return 1
    else:
        return -1

# data = np.array([[1,2,3],[2,3,4],[4,5,6]])
# label = np.array([[1], [-1], [1]])

# adaboost_stump = (data, label, 3)
# print(adaboost_predict(adaboost_stump, np.array([2,3,4])))

import pandas as pd
data_df = pd.read_csv('sonar.csv', header='None')

# data_df['60'] = data_df['60'].map({'M':1, 'R':-1})
data_array = data_df.values()
data = data_array(:,,:-1)
label = data_array[:,-1]
