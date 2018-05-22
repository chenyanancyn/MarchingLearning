# import numpy as np
# import pandas as pd

# # train
# train_df = pd.read_csv('./ml-100k/u2.base', header=None, index_col=None)
# # print(type(train_data))
# train_data = train_df.values
# # print(train_data)   # numpy
# # print(train_data[0][0].split('\t'))  # ['1', '3', '4', '878542960']

# # test
# test_df = pd.read_csv('./ml-100k/u2.test')
# test_data = test_df.values


# # train
# u2_items = {}
# for data in train_data:
#     sample = data[0].split('\t')
#     if u2_items.get(sample[0]) is None:
#         u2_items[sample[0]] = []
#     u2_items[sample[0]].append(sample[1])
    
# # print(u2_items)

# # print(set(u2_items['1']) & set(u2_items['2']))
# # print(len(set(u2_items['1']) & set(u2_items['2'])))


# def user_similarity(train):
#     import math
    
#     w = dict()
#     for u in train.keys():
#         for v in train.keys():
#             if u == v:
#                 continue
#             uv = u + v
#             w[uv] = len(set(u2_items[u]) & set(u2_items[v]))
#             w[uv] = w[uv] / math.sqrt(len(train[u]) * len(train[v]))
            
#     return w

# # print(user_similarity(u2_items))

# # 共有1682部电影
# item_u2s = {}
# for i in range(1, 1683):
#     if i % 100 == 0:
#         print(i)
#     key = str(i)
#     for data in train_data:
#         sample = data[0].split('\t')
#         if sample[1] == key:
#             if item_u2s.get(key) is None:
#                 item_u2s[key] = []
#             item_u2s[key].append(sample[0])

# # print(item_u2s)

d = {1: 2, 3: 4}
if 3 in d:
    print(3)