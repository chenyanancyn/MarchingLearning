import numpy as np


# 串行方法 -- boosting族 ： Adaboost, GBDT, XGBoost
# 1.Boosting集成方法
# boosting方法产生于计算学习理论（Computational Learning Theory）
# 1.根据当前的数据训练出一个弱模型
# 2.根据该弱模型的表现调整数据样本的权重，具体而言：
#                  让该模型做错的样本在后续的训练中获得更多的关注
#                  让该样本做对的样本在后续的训练中获得较少的关注
# 3.最后再根据弱模型的表现决定该弱模型的“话语权”，亦即投票表决时的“可信度”。自然，表现越好的就越具有话语权。

# 2.Adaboost算法
# 由boosting方法的陈述可知，问题的关键在于两点：
# 1.如何根据弱模型的表现更新训练集的权重    对每个样本的作用
# 2.如何根据弱模型的表现决定弱模型的话语权  整体价值体现

# Adaboost算法：
# 采取了加权错误率的方法更新样本的权重用来解决二分类问题，标签是{-1, 1}
# 弱分类器选择决策树桩，决策树桩，决策树桩是单层二叉树，以加权错误率作为分割标准
# 假设现有的二分类训练数据集：D = {(x1, y1), (x2, y2), ..., (xn, yn)}
# 其中每个样本由特征x和类别y组成，且：xi属于X 包含于Rn；yi属于Y={-1, +1}

# Adaboost算法步骤：
# 以什么最为分类标准：加权错误率
# 二叉树还是多叉树：二分类，二叉树
# 什么时候终止：一层就停止

# 2.1 单层决策树
# treeStump函数实现单层决策树 
# 使用index和value，将数据集分为两部分：两种：左正或右正
# 使用两个函数实现单层决策树:
# split函数实现对data的分割
# def split(data, index, value, lr):  # data不包含标签列, 要求data是mat, 输出的预测pre是列向量
#     data = np.array(data)
#     pred_label = []
#     if lr == 'l':
#         for d in data:
#             if d[index] > value:   # 分割值左侧为正样本 
#                 pred_label.append(-1)
#             else:
#                 pred_label.append(1)
#     else:
#         for d in data:
#             if d[index] > value:
#                 pred_label.append(1)
#             else:
#                 pred_label.append(-1)

#     return pred_label

def split(data, index, value, lr):  # lr='l'或lr='r',要求data是mat, pre长度和data列的长度是一样的
    data = np.array(data)
    m, n = data.shape
    # 首先创建pred_labeld为全1的列向量
    pred_label = np.ones((m, 1))
    if lr == 'l':
        pred_label[data[:, index] > value] = -1    #**********布尔值索引
    else:
        pred_label[data[:, index] <= value] = -1

    return pred_label

# a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# print(split(a, 1, 5, 'r'))

# 单层决策树建树
def tree_stump(data, labelList, w):
    data = np.array(data)
    m, n = data.shape
    labelList = labelList.reshape((m, 1))
    w = w.reshape((m, 1))  # 权值分布
    min_w_err = 1e+8  # 定义最小加权错误率
    b_pre = np.zeros((m, 1))  # 最优预测结果
    stump = {'value': -1, 'index': -1, 'lr': 'l'}
    for index in range(n):  # 取出data的每一列
        fea = set(data[:, index])   # 取出一列的值,即该列的所有二分标准备选值
        for value in fea:
            for lr in ['l', 'r']:   # ++ 因为是决策树桩,所以只有一次比较,就出结果,因为不确定真实标签是-1,还是+1,
                                    # 所以就只能分为两类,即真实标签中,当样本的某个特征大于某个值时,标签为+1,
                                    # 或者当样本的某个特征小于某个值时,标签为+1,此为两种结果,故要分为两个种类
                pred_label = split(data, index, value, lr)
                err = np.zeros((m, 1))
                err[pred_label != labelList] = 1
                w_err = np.matmul(err.T, w)  # 求内积 ****** multiply 求array对应相乘
                if w_err < min_w_err:
                    min_w_err = w_err
                    stump['index'] = index
                    stump['value'] = value
                    stump['lr'] = lr
                    b_pre = pred_label

    return  min_w_err, b_pre, stump


# 2.2 树桩"话语权"与更新权重
def adaboost(data, label,  n_stump):
    data = np.array(data)
    m, n = data.shape
    label = label.reshape((m, 1))
    pred_label = np.zeros((m, 1))   # 最后的预测标签,放入此列表中
    w = np.ones((m, 1)) / m  
    boost_stump = {}
    for n in range(n_stump):
        w_err, b_pre, stump = tree_stump(data, label, w)
        alpha = 0.5 * np.log((1 - w_err) / max(w_err, 1e-8) ) # 加权错误率的"话语权", 即每个树桩对应的权重

        e_ayg = np.exp(-alpha * np.multiply(label, b_pre))
        w = np.multiply(w, e_ayg)
        w = w / np.sum(w)
        boost_stump[n] = (alpha, stump)

        #####算法流程完成,检测当前效果
        pred_label += b_pre*alpha
        label_current = np.ones((m, 1))
        label_current[pred_label < 0] = -1   # 相当于最后的內个阶越函数
        print((label_current == label).flatten().tolist().count(True) / m)
        print(alpha, stump)
        #####算法流程完成,检测当前的效果
    
    return boost_stump



import pandas as pd
data_df = pd.read_csv('sonar.csv', header=None)

data_df[60] = data_df[60].map({'M': 1,'R': -1})
# print(data_df.head(5))

data_array = data_df.values

data = data_array[:, :-1]
label = data_array[:, -1]

adaboost_stump=adaboost(data,label,20)