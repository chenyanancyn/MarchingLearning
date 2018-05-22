import numpy as np
import pandas as pd


# 读取数据集
# train
train_df = pd.read_csv('./ml-100k/u2.base', header=None, index_col=None)
# print(type(train_data))
train_df = train_df.values
# print(train_data)   # numpy
# print(train_data[0][0].split('\t'))  # ['1', '3', '4', '878542960']

# test
test_df = pd.read_csv('./ml-100k/u2.test', header=None, index_col=None)
test_df = test_df.values

# 整理数据集   943 users on 1682 movies.
# 矩阵：横坐标表示对应一个用户给一个电影打的分数， 纵坐标表示用户id   
# train  
train_data = np.zeros((943, 1682), dtype=np.int)   # 因为矩阵索引从0开始
for data in train_df:
    infor = data[0].split('\t')
    user = int(infor[0]) - 1
    movie = int(infor[1]) - 1
    train_data[user][movie] = 1
# print(train_data)
# test
test_data = np.zeros((944, 1683))   # 因为矩阵索引从0开始
for data in test_df:
    infor = data[0].split('\t')
    user = int(infor[0])
    movie = int(infor[1])
    test_data[user][movie] = 1


def cos_sim(x, y):
    """余弦相似性

    Args:
    - x: mat, 以行向量的形式存储
    - y: mat, 以行向量的形式存储

    :return: x 和 y 之间的余弦相似度
    """
    numerator = np.matmul(x, y.T)  # x 和 y 之间的内积
    denominator = np.sqrt(np.matmul(x, x.T)) * np.sqrt(np.matmul(y, y.T))
    return (numerator / max(denominator, 1e-7))

# print(cos_sim(train_data[0], train_data[1]))


# 对于任意矩阵，计算任意两个行向量之间的相似度：
def similarity(data):
    """计算矩阵中任意两行之间的相似度
    Args:
    - data: mat, 任意矩阵

    :return: w, mat, 任意两行之间的相似度
    """

    m = np.shape(data)[0]  # 用户的数量
    # 初始化相似矩阵
    w = np.mat(np.zeros((m, m)))   # 相似度矩阵w是一个对称矩阵，而且在相似度矩阵中，约定自身的相似度的值为 $0$ 

    for i in range(m):
        for j in range(i, m):
            if not j == i:
                # 计算任意两行之间的相似度
                w[i, j] = cos_sim(data[i], data[j])
                w[j, i] = w[i, j]
            else:
                w[i, j] = 0
    return w


def user_based_recommend(data, w, user):
    """基于用户相似性为用户 user 推荐物品

    Args:
    - data: mat, 用户物品矩阵
    - w: mat, 用户之间的相似度
    - user: int, 用户编号

    :return: predict, list, 推荐列表
    """
    m, n = np.shape(data)
    interaction = data[user, ]  # 用户 user 与物品信息

    # 找到用户 user 没有互动过的物品
    not_inter = []
    for i in range(n):
        if interaction[0, i] == 0:  # 没有互动的物品
            not_inter.append(i)

    # 对没有互动过的物品进行预测
    predict = {}
    for x in not_inter:
        item = np.copy(data[:, x])  # 找到所有用户对商品 x 的互动信息
        for i in range(m):  # 对每一个用户
            if item[i, 0] != 0:
                if x not in predict:
                    predict[x] = w[user, i] * item[i, 0]
                else:
                    predict[x] = predict[x] + w[user, i] * item[i, 0]
    return sorted(predict.items(), key=lambda d: d[1], reverse=True)


# 如果是 TOP N 推荐，为用户推荐前N个打分最高的物品：

def top_k(predict, n):
    """为用户推荐前 n 个物品

    Args:
    - predict: list, 排好序的物品列表
    - k: int, 推荐的物品个数

    :return: top_recom, list, top n 个物品
    """
    top_recom = []
    len_result = len(predict)
    if n >= len_result:
        top_recom = predict
    else:
        for i in range(n):
            top_recom.append(predict[i])
    return top_recom
